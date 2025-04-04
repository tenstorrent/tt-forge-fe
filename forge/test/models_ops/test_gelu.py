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


class Gelu0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, gelu_input_0):
        gelu_output_1 = forge.op.Gelu("", gelu_input_0, approximate="none")
        return gelu_output_1


class Gelu1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, gelu_input_0):
        gelu_output_1 = forge.op.Gelu("", gelu_input_0, approximate="tanh")
        return gelu_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Gelu0,
        [((1, 6, 3072), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 384, 4096), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 128, 3072), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 128, 3072), torch.float32)],
        {
            "model_name": ["pt_albert_base_v2_token_cls_hf", "pt_albert_base_v2_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu0,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 13, 1536), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 197, 3072), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 197, 4096), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((2, 1, 8192), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((2, 1, 4096), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((2, 1, 6144), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 384, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 384, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1500, 1536), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1, 1536), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1280, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1280, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1500, 5120), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1, 5120), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 768, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 768, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1500, 3072), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1, 3072), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1024, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1024, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1500, 4096), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1, 4096), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 512, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 512, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1500, 2048), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1, 2048), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 1, 2048), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu0,
        [((1, 2, 5120), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 576, 4096), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 204, 3072), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"approximate": '"none"'}},
    ),
    (
        Gelu0,
        [((1, 201, 3072), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1536), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 128, 8192), torch.float32)],
        {
            "model_name": ["pt_albert_xlarge_v1_mlm_hf", "pt_albert_xlarge_v1_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 128, 8192), torch.float32)],
        {
            "model_name": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v2_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu0,
        [((1, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": ["pt_albert_large_v2_mlm_hf", "pt_albert_large_v2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu0,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 14, 3072), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 9, 3072), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 128, 16384), torch.float32)],
        {
            "model_name": ["pt_albert_xxlarge_v2_token_cls_hf", "pt_albert_xxlarge_v2_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu0,
        [((1, 128, 16384), torch.float32)],
        {
            "model_name": ["pt_albert_xxlarge_v1_token_cls_hf", "pt_albert_xxlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 256, 4096), torch.float32)],
        {
            "model_name": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_xglm_facebook_xglm_564m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu0,
        [((1, 384, 3072), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 6, 18176), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 207, 14336), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu1,
        [((1, 7, 16384), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu1,
        [((1, 207, 9216), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu1,
        [((1, 107, 16384), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu1,
        [((1, 107, 24576), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu1,
        [((1, 7, 3072), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 256, 3072), torch.float32)],
        {
            "model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 32, 8192), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 256, 10240), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 32, 3072), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 32, 10240), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 256, 8192), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu0,
        [((1, 256, 8192), torch.float32)],
        {"model_name": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99, "op_params": {"approximate": '"none"'}},
    ),
    (
        Gelu0,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 1, 1024), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu0,
        [((1, 256, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 2048, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 11, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu1,
        [((1, 12, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"tanh"'},
        },
    ),
    (
        Gelu0,
        [((1, 3072, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu1,
        [((1, 61, 2048), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu1,
        [((1, 61, 2816), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu1,
        [((1, 1, 2816), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu1,
        [((1, 61, 1024), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"approximate": '"tanh"'}},
    ),
    (
        Gelu0,
        [((1, 197, 1536), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 197, 768), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 19200, 256), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 4800, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1200, 1280), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 300, 2048), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 768, 384), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 196, 3072), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1024, 512), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mlp_mixer_base_img_cls_github",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 196, 4096), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 256, 256), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"approximate": '"none"'}},
    ),
    (
        Gelu0,
        [((1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 4096, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 16384, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 1024, 640), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 56, 56, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 28, 28, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 14, 14, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 7, 7, 3072), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 56, 56, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"approximate": '"none"'}},
    ),
    (
        Gelu0,
        [((1, 28, 28, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"approximate": '"none"'}},
    ),
    (
        Gelu0,
        [((1, 14, 14, 2048), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"approximate": '"none"'}},
    ),
    (
        Gelu0,
        [((1, 7, 7, 4096), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"approximate": '"none"'}},
    ),
    (
        Gelu0,
        [((1, 3136, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 784, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 196, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
    (
        Gelu0,
        [((1, 49, 3072), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"approximate": '"none"'},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()

    forge_property_recorder.record_forge_op_name("Gelu")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_name":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "op_params":
            forge_property_recorder.record_forge_op_args(metadata_value)
        else:
            print("there is no utility function available to record these details")

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

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
