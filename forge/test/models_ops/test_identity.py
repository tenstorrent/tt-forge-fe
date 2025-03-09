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
import pytest


class Identity0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, identity_input_0):
        identity_output_1 = forge.op.Identity("", identity_input_0)
        return identity_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Identity0,
        [((2, 1, 2048), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((64, 1, 1), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 13, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((2, 12, 13, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((2, 13, 3072), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((64, 1, 13), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 8192), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 1536), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((48, 1, 1), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((48, 1, 13), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 6144), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 1024), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((32, 1, 1), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((32, 1, 13), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 4096), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1500, 1024), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 1500, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 4096), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 1, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 4096), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 20, 1, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 20, 1500, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 5120), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 20, 1, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 5120), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1500, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 1500, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 1536), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 1, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 1536), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1500, 512), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 1500, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 2048), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 1, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 2048), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1500, 768), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 1500, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 3072), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 1, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 3072), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 2, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 20, 2, 2), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 20, 2, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 2, 5120), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((16, 7, 7), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 39, 39), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf", "DeepSeekWrapper_decoder"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 204, 204), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 204, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 201, 201), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 201, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 128, 128), torch.float32)],
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
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((16, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 4096), torch.float32)],
        {
            "model_name": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_xglm_facebook_xglm_564m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 384, 1024), torch.float32)],
        {"model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 384, 384), torch.float32)],
        {"model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 384, 768), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 384, 384), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 71, 6, 6), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 4544), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 8, 10, 10), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 64, 334, 334), torch.float32)], {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (Identity0, [((1, 334, 4096), torch.float32)], {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (Identity0, [((1, 8, 7, 7), torch.float32)], {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 256, 256), torch.float32)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 256, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 20, 256, 256), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 32, 32), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 2560), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 20, 32, 32), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 32, 32), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 4, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 128, 128), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 7, 768), torch.float32)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 7, 7), torch.float32)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (Identity0, [((32, 256, 256), torch.float32)], {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (Identity0, [((256, 2048), torch.float32)], {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((32, 32, 32), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((32, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((16, 32, 32), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((32, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((12, 32, 32), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((32, 768), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (Identity0, [((12, 256, 256), torch.float32)], {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
    (Identity0, [((256, 768), torch.float32)], {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
    (Identity0, [((256, 1024), torch.float32)], {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 12, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 11, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 13, 13), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 13, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 5, 5), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 5, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 16, 6, 6), torch.float32)], {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 16, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 28, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 14, 35, 35), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 39, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 28, 39, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 28, 29, 29), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 16, 39, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 14, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 14, 39, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 768, 128), torch.float32)],
        {"model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 61, 1024), torch.float32)],
        {
            "model_name": [
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 61, 2816), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 2816), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 61, 4096), torch.float32)], {"model_name": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 61, 512), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 8, 61, 61), torch.float32)], {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 61, 2048), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 8, 1, 61), torch.float32)], {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 6, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 61, 768), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 61, 3072), torch.float32)], {"model_name": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 12, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 256, 8192), torch.float32)], {"model_name": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 9216), torch.float32)],
        {"model_name": ["pt_alexnet_alexnet_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 4096), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 197, 768), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 197, 197), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 197, 192), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 3, 197, 197), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 197, 384), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 197, 197), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1280), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1792), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768, 384), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768, 49), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 3072), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 512, 256), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm", "pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 512, 196), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 196, 2048), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 196, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 512, 49), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 2048), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 768, 196), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 196, 3072), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 196, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 512), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 49), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 4096), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 1024), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1024, 196), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 196, 4096), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 196, 1024), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 28, 28), torch.float32)],
        {"model_name": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 512, 3025), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 512, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1, 512, 50176), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16384, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16384, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 640), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 4096, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 3136, 96), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((64, 3, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((64, 49, 96), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((16, 6, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((16, 49, 192), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 784, 192), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((4, 12, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((4, 49, 384), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 196, 384), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 24, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 4096, 1, 1), torch.float32)], {"model_name": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 197, 1024), torch.float32)],
        {"model_name": ["pt_vit_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 197, 197), torch.float32)],
        {"model_name": ["pt_vit_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_forge_property):
    record_forge_property("tags.op_name", "Identity")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        record_forge_property("tags." + str(metadata_name), metadata_value)

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
