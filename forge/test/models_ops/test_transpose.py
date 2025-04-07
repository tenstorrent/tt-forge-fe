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


class Transpose0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-3, dim1=-2)
        return transpose_output_1


class Transpose1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-2, dim1=-1)
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


class Transpose4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-5, dim1=-4)
        return transpose_output_1


class Transpose5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-4, dim1=-2)
        return transpose_output_1


class Transpose6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-5, dim1=-3)
        return transpose_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Transpose0,
        [((1, 6, 12, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 6), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 6, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
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
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 384, 16, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 384, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 384, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 384), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 384, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 2), torch.float32)],
        {
            "model_name": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
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
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 128), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
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
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 128, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
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
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 13, 12, 32), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 13, 32), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 13, 32), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 32, 13), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 13, 32), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 384), torch.float32)],
        {
            "model_name": [
                "onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 100, 8, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 100, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 100), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 100, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 280), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 280, 8, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 280, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 280), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 280, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 280, 256), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2048), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 196), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 12, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 768), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 196), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 16, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1024), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_gemma_google_gemma_2b_text_gen_hf",
                "pt_gemma_google_gemma_1_1_2b_it_qa_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 1, 32, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 64, 1), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 32, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((2, 13, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 13, 64), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((13, 13, 12), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 13, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 12, 13, 64), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 12, 13, 64), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 64, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
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
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 3072), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
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
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 13, 32, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 13, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 64, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8192, 2048), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 8192), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 1, 16, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 64, 1), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 16, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1024, 768), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 13, 16, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 13, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 64, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 1024), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 4096), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 1536), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 1, 24, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((48, 64, 1), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 24, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1536, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 13, 24, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 13, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((48, 64, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6144, 1536), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 6144), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 384, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 6, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 384), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 1536), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51865, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 1280), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 20, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 20, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 64, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 20, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 1500, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 20, 1500, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 64, 1500), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5120, 1280), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 5120), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51865, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 12, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51865, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 16, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51865, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 512), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_nbeats_generic_basis_clm_hf",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 8, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 8, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 512), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "pt_t5_t5_small_text_gen_hf",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 2048), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51865, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 20, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 2, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 20, 2, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 2, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 64, 2), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51866, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 7, 8, 64), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 7, 64), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 7), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 8, 7, 64), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 588, 16, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 588), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 588, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 588), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5504, 2048), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 5504), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32256, 2048), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 4096), torch.float32)],
        {
            "model_name": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 39, 32, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 39), torch.float32)],
        {
            "model_name": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 39, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 39, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 39, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 128, 39), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((11008, 4096), torch.float32)],
        {
            "model_name": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 11008), torch.float32)],
        {
            "model_name": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((102400, 4096), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 576), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 577, 16, 64), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 577, 64), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 577), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 577, 64), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 596, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 596), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 596, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 596, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 596, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 128, 596), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32064, 4096), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 204, 12, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 204, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 204, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 204, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 204), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 201, 12, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 201, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 201, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 201, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 201), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3129, 1536), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128, 2048), torch.float32)],
        {
            "model_name": ["pt_albert_xlarge_v1_mlm_hf", "pt_albert_xlarge_v2_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((30000, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 1024), torch.float32)],
        {
            "model_name": ["pt_albert_large_v2_mlm_hf", "pt_albert_large_v1_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 14, 12, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 14, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 14, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 14, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 14), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 768), torch.float32)],
        {
            "model_name": ["pt_albert_base_v1_mlm_hf", "pt_albert_base_v2_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 9, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 9), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 64, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16384, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 16384), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 1024), torch.float32)],
        {
            "model_name": ["pt_albert_large_v2_token_cls_hf", "pt_albert_large_v1_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 4096), torch.float32)],
        {
            "model_name": ["pt_albert_xxlarge_v1_mlm_hf", "pt_albert_xxlarge_v2_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((9, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((30522, 768), torch.float32)],
        {
            "model_name": ["pt_bert_bert_base_uncased_mlm_hf", "pt_distilbert_distilbert_base_uncased_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4608, 1536), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 16, 96), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 32, 96), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 96, 32), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 32, 96), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((250880, 1536), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51200, 1024), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 384, 12, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 384, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 384, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 384, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 384), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((119547, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_multilingual_cased_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((9, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((28996, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 10, 8, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 10), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 2048), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf", "pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 10, 4, 256), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((131072, 2048), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((18176, 4544), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4544, 18176), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4672, 4544), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 71, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 6), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf", "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((4544, 4544), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((65024, 4544), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 3072), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 10, 12, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1024, 3072), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 256, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((9216, 3072), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 9216), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((131072, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((23040, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 23040), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12288, 4096), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 334, 64, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 16, 334), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((64, 334, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 64, 334, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 64, 334, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 64, 334), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((4096, 3584), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 207, 16, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 128, 207), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf", "pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 3584), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 207, 8, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf", "pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 16, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 256, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((3584, 4096), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((14336, 3584), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((3584, 14336), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256000, 3584), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 7, 8, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 7), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gemma_google_gemma_2b_text_gen_hf",
                "pt_gemma_google_gemma_1_1_2b_it_qa_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 7, 1, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 7, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 7, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 7, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 7), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16384, 2048), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf", "pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 16384), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf", "pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256000, 2048), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf", "pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 2304), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1024, 2304), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 207, 4, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 8, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 8, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 256, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2304, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((9216, 2304), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2304, 9216), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256000, 2304), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 107, 8, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 107, 1, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 3072), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 107, 16, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 256, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 4096), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((24576, 3072), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 24576), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256000, 3072), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 2304), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 7, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 7, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 7, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 7, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 7), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 256, 64), torch.float32)],
        {
            "model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((50257, 768), torch.float32)],
        {
            "model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 16, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2560, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 20, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 20, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 128, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((10240, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2560, 10240), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((50257, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 32), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 20, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 20, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 128, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 16, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((50257, 2048), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 32, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 128, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((14336, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 14336), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 32, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 8, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 128, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 24, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 24, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 128, 4), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8192, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf", "pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 8192), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 3072), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 8, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 64, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 24, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf", "pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 8, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 24, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 128, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128256, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 128, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32000, 4096), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf", "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 32, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 8, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 64, 32), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1024, 512), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 16, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 32), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 512), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((50272, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 512), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((50272, 512), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((50272, 768), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 256, 224, 224), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 224, 256, 224), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 256), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 50176, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 50176, 1, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 50176, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 50176), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 512, 8, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 512, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 512, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 512, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 128, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 512, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((322, 1024), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 64, 55, 55), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 55, 64, 55), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((322, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3025, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3025, 1, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 3025, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 322, 3025), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((261, 1024), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 224, 3, 224), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((261, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 50176, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 50176, 1, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 50176, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 261, 50176), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2048, 8, 32), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 32), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 2048, 32), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2048, 8, 160), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 2048, 160), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 160, 2048), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 160), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 160), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 160), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 160, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 96), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 96), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 96, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 2048, 96), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((262, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 32, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 256), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 256, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 80, 256), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51200, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 11, 32, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 11, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 80, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 32, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 12, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 12, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 12, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 80, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 32, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 48, 5), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 5, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 5, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 5, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 96, 5), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8192, 3072), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 13, 32, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 48, 13), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 13, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 13, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 13, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 96, 13), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 32, 96), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 48, 256), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 256, 96), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256, 96), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 256, 96), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 96, 256), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32064, 3072), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 29, 16, 64), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 29), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 29, 64), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 29), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2816, 1024), torch.float32)],
        {
            "model_name": [
                "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 2816), torch.float32)],
        {
            "model_name": [
                "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((151936, 1024), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 16, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 6, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 16, 6, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 6, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 64, 6), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 35, 12, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 2, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8960, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 8960), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((151936, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3584, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_7b_token_cls_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 28, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_7b_token_cls_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((28, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 28, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 28, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((28, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((18944, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_7b_token_cls_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3584, 18944), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_7b_token_cls_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((152064, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((896, 896), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 14, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 35), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 896), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 2, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((14, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((14, 64, 35), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4864, 896), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((896, 4864), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((151936, 896), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((11008, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 11008), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((151936, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 39, 16, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 39, 2, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 29, 16, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 29), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 29, 2, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 16, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 29, 12, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 12, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 39, 12, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 128, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 39, 14, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 39, 2, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((14, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 14, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 14, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((14, 64, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 39, 28, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 39, 4, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((28, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 28, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 28, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((28, 128, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 29, 14, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose0,
        [((1, 29, 2, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((14, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 14, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 14, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((14, 64, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 13, 28, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 13), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 13, 4, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((28, 13, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 28, 13, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 28, 13, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((28, 128, 13), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 29, 28, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose0,
        [((1, 29, 4, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((28, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 28, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 28, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((28, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((250002, 768), torch.float32)],
        {"model_name": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((3, 768), torch.float32)],
        {
            "model_name": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 12), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 1, 1), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 61, 12, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((61, 61, 12), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 2048), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32128, 768), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 8), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 1, 1), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 61, 8, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((61, 61, 8), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 8, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 8, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32128, 512), torch.float32)],
        {
            "model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 16), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 1, 1), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 61, 16, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((61, 61, 16), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32128, 1024), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 512), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 6), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 1, 1), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 384), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 61, 6, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((61, 61, 6), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256008, 2048), torch.float32)],
        {"model_name": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256008, 1024), torch.float32)],
        {"model_name": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((1, 1, 1024, 72), torch.float32)],
        {
            "model_name": [
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((512, 72), torch.float32)],
        {"model_name": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((96, 512), torch.float32)],
        {"model_name": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256, 72), torch.float32)],
        {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 256), torch.float32)],
        {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2048, 72), torch.float32)],
        {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((48, 2048), torch.float32)],
        {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((4096, 9216), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 4096), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 784), torch.float32)],
        {
            "model_name": ["pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 128), torch.float32)],
        {
            "model_name": ["pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 64), torch.float32)],
        {
            "model_name": ["pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 12), torch.float32)],
        {
            "model_name": ["pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 3), torch.float32)],
        {
            "model_name": ["pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 12), torch.float32)],
        {
            "model_name": ["pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 64), torch.float32)],
        {
            "model_name": ["pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((784, 128), torch.float32)],
        {
            "model_name": ["pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 27, 27, 12), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 27, 27), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 27, 27), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 27, 12, 27), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((197, 197, 12), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 197, 197), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 27, 27, 16), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 27, 27), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 27, 27), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 27, 16, 27), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((197, 197, 16), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 197, 197), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 384, 196), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 6, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 384), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 192, 196), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 192), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 3, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3, 64, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 192), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 768), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2208), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1920), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet201_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((18, 1024), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet121_hf_xray_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1664), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1280), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1792), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 19200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 64), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 19200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 1, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 64), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 19200, 256), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 19200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 256), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 120, 160, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 160, 120), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 4800), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 128), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4800, 2, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4800, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 2, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4800, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 128), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4800, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 4800), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 512), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 60, 80, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 80, 60), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 1200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((320, 320), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1200, 5, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1200, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 5, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1280, 320), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1200, 1280), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((320, 1280), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 30, 40, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 40, 30), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 8, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 300, 2048), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 15, 20, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 20, 15), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1536), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 196), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((196, 384), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((11221, 768), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 1024), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 196), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((196, 512), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((21843, 768), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 3, 16, 16, 16, 16), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"dim0": "-5", "dim1": "-4"}},
    ),
    (
        Transpose5,
        [((1, 16, 3, 16, 16, 16), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 16, 16, 16, 3, 16), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((512, 768), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256, 512), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 256), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 256, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 512), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 9216), torch.float32)],
        {"model_name": ["pt_mnist_base_img_cls_github"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((10, 128), torch.float32)],
        {"model_name": ["pt_mnist_base_img_cls_github"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1001, 768), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1001, 1024), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1001, 1280), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 576), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 960), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1088), torch.float32)],
        {
            "model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 7392), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 888), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 3712), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 440), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2520), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1008), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 912), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 672), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2016), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 784), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1512), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 400), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 3024), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 16384), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 1, 64), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 256, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
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
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 16384), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 128, 128, 64), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 128, 128), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 4096), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4096, 2, 64), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 128), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 256), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 2, 64), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 256, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 256, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 64, 256), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4096, 64), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 4096), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 64, 64, 128), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 1024), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1024, 5, 64), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 320), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 256), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 5, 64), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5, 256, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 256, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 64, 256), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1024, 64), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1024), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 32, 32, 320), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 32, 32), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 256), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 64), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 64), torch.float32)],
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
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 256), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048, 256), torch.float32)],
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
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 320), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 128), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 1, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 128, 128, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4096, 2, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 2, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4096, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 64, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((160, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1024, 5, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 5, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1024, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((640, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 640), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 640, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((160, 640), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 32, 32, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_swin_swin_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 256), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 16, 16, 256), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 16, 16), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 160), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 32), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 56, 96, 56), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 8, 7, 8, 7, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((288, 96), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose6,
        [((64, 49, 3, 3, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose3,
        [((3, 49, 64, 3, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 64, 49, 3, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((49, 49, 3), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 3, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 3, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 8, 8, 7, 7, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((384, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 4, 7, 4, 7, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((576, 192), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose6,
        [((16, 49, 3, 6, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose3,
        [((3, 49, 16, 6, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 16, 49, 6, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((96, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((49, 49, 6), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 6, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((16, 6, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((96, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 4, 4, 7, 7, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((384, 768), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 2, 7, 2, 7, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((1152, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose6,
        [((4, 49, 3, 12, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose3,
        [((3, 49, 4, 12, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 4, 49, 12, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((49, 49, 12), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4, 12, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 12, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 2, 2, 7, 7, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((768, 1536), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 1, 7, 1, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((2304, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose6,
        [((1, 49, 3, 24, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose3,
        [((3, 49, 1, 24, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 49, 24, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((49, 49, 24), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((24, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 24, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 1, 1, 7, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose2,
        [((1, 7, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 7, 7), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 56, 56), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 56, 128, 56), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((1, 8, 7, 8, 7, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((384, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose6,
        [((64, 49, 3, 4, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose3,
        [((3, 49, 64, 4, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 64, 49, 4, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((49, 49, 4), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((4, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((64, 4, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 4, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((1, 8, 8, 7, 7, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose3,
        [((1, 4, 7, 4, 7, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((768, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose6,
        [((16, 49, 3, 8, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose3,
        [((3, 49, 16, 8, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 16, 49, 8, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((49, 49, 8), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 8, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((16, 8, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((1, 4, 4, 7, 7, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose3,
        [((1, 2, 7, 2, 7, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((1536, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose6,
        [((4, 49, 3, 16, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose3,
        [((3, 49, 4, 16, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 4, 49, 16, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((49, 49, 16), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((4, 16, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((4, 16, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((1, 2, 2, 7, 7, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose3,
        [((1, 1, 7, 1, 7, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose6,
        [((1, 49, 3, 32, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose3,
        [((3, 49, 1, 32, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 1, 49, 32, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((49, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((32, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((1, 1, 1, 7, 7, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose2,
        [((1, 7, 7, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 1024, 7, 7), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 96, 3136), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 49, 3, 32), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((16, 49, 6, 32), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((4, 49, 12, 32), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 49, 24, 32), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 25088), torch.float32)],
        {
            "model_name": [
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 3600), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 900), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 225), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 1600), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 400), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 100), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 6400), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 25600), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4, 5880), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 80, 5880), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 17, 4480), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 17, 1120), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 17, 280), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 85, 8400), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 85, 3549), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"dim0": "-2", "dim1": "-1"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Transpose")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        forge_property_recorder("tags." + str(metadata_name), metadata_value)

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
