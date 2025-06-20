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
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_bert_base_japanese_seq_cls_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 768), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 7, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 61, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 61, 3072), torch.float32)], {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 12, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 3072), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_alexnet_base_img_cls_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_alexnet_base_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1, 512), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1500, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 1500, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 2048), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 1, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 2048), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 11, 11), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 9, 9), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 128, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
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
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 588, 588), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1536), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 7, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 28, 28), torch.bfloat16)],
        {"model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((32, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 256, 2560), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 14, 39, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 39, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 2, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 4096, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 5, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 8, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 50, 1024), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 50, 50), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 50, 4096), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1024, 1, 1), torch.float32)],
        {"model_names": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Identity0, [((1, 1024), torch.float32)], {"model_names": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 1536), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 197, 384), torch.bfloat16)],
        {"model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 197, 197), torch.bfloat16)],
        {"model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((8, 100, 100), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 100, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 850, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((8, 850, 850), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 850, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((8, 100, 850), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 100, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Identity0, [((1, 512, 1, 1), torch.bfloat16)], {"model_names": ["pt_dla_dla34_in1k_img_cls_timm"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 257, 768), torch.bfloat16)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 257, 257), torch.bfloat16)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 257, 3072), torch.bfloat16)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1024, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 196, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 196, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1, 512, 3025), torch.bfloat16)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 512, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1, 1, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 12, 12), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 768, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 10, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 8, 8), torch.float32)],
        {
            "model_names": [
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1408), torch.bfloat16)],
        {"model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 2304), torch.bfloat16)],
        {"model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 19200, 300), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 19200, 64), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 19200, 256), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 2, 4800, 300), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 4800, 128), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 4800, 512), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 5, 1200, 300), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1200, 320), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1200, 1280), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 300, 300), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 300, 512), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 300, 2048), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 196, 3072), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 196, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1, 512, 50176), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 35, 35), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((64, 3, 49, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((64, 49, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 56, 56, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 56, 56, 96), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((16, 6, 49, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((16, 49, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 28, 28, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 28, 28, 192), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((4, 12, 49, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((4, 49, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 14, 14, 1536), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 14, 14, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 24, 49, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 49, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 7, 7, 3072), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 7, 7, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 61, 512), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 61, 1024), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 6, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 1024), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((16, 256, 256), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 256, 4096), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_bart_facebook_bart_large_mnli_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 14, 128), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 14, 14), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf", "pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 14, 768), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf", "pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768, 49), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 3072), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (Identity0, [((1, 64, 12, 12), torch.bfloat16)], {"model_names": ["pt_mnist_base_img_cls_github"], "pcc": 0.99}),
    (Identity0, [((1, 128), torch.bfloat16)], {"model_names": ["pt_mnist_base_img_cls_github"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Identity0, [((256, 2048), torch.float32)], {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 32, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 2048), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((32, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((64, 4, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((64, 49, 128), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 56, 56, 512), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 56, 56, 128), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((16, 8, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((16, 49, 256), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 28, 28, 1024), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 28, 28, 256), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((4, 16, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Identity0, [((4, 49, 512), torch.bfloat16)], {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 14, 14, 2048), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 14, 14, 512), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 1024), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision", "pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 7, 7, 4096), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 7, 7, 1024), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 1500, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 1536), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 1, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 1536), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 256, 8192), torch.float32)], {"model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 8, 12, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 240), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((2, 8, 7, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 197, 192), torch.bfloat16)],
        {"model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 3, 197, 197), torch.bfloat16)],
        {"model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 4, 4), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 49), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 4096), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 512, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm", "pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 512, 49), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 2048), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 49, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm", "pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 14, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 1024), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 16, 1, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 13, 768), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((2, 12, 13, 13), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((2, 13, 3072), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((2, 16, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 4096), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 61, 2816), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1, 2816), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 4096, 1, 1), torch.bfloat16)], {"model_names": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 197, 4096), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 2, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_v3_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 20, 2, 2), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_v3_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 20, 2, 1500), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_v3_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 2, 5120), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_v3_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 15, 15), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1280), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 9, 128), torch.float32)],
        {"model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 384, 384), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Identity0, [((256, 768), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 14, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 8, 61, 61), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 61, 2048), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 8, 1, 61), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 12, 201, 201), torch.bfloat16)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 201, 768), torch.bfloat16)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 50, 768), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 50, 50), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 50, 3072), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 16, 16), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 2560), torch.bfloat16)],
        {"model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16384, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16384, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1024, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 256, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 768, 128), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((64, 4, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((64, 64, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 64, 64, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 64, 64, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((16, 8, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((16, 64, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 32, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 32, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((4, 16, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((4, 64, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 16, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 16, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 64, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 8, 4096), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((64, 3, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 64, 64, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((16, 6, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((16, 64, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 32, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((4, 12, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((4, 64, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 16, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 24, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 8, 8, 3072), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 8, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 204, 204), torch.bfloat16)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 204, 768), torch.bfloat16)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1370, 1280), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 1370, 1370), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1370, 5120), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 20, 1500, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 5120), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 1500, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 3072), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 1, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1445, 192), torch.bfloat16)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 3, 1445, 1445), torch.bfloat16)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 9216), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (Identity0, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 1792), torch.bfloat16)],
        {"model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 522, 522), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 207, 207), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 512, 196), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 196, 2048), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 196, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 197, 3072), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 24, 44, 44), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 39, 39), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 71, 6, 6), torch.float32)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 6, 4544), torch.float32)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (Identity0, [((1, 64, 334, 334), torch.float32)], {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (Identity0, [((1, 334, 4096), torch.float32)], {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 32, 2560), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 20, 32, 32), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 11, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 11, 11), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 12, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 5, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 13, 13), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 13, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 40, 256, 256), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 256, 5120), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 40, 12, 12), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 5120), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 35, 35), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 28, 35, 35), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 28, 13, 13), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 13, 3584), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 2048), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 32, 1, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 32, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 8192), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 107, 107), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 107, 107), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 24, 4, 4), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 32, 10, 10), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 8, 8), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 135, 135), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 32, 128, 128), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 256, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Identity0, [((1, 40, 6, 6), torch.float32)], {"model_names": ["pt_phi4_microsoft_phi_4_clm_hf"], "pcc": 0.99}),
    (Identity0, [((1, 6, 5120), torch.float32)], {"model_names": ["pt_phi4_microsoft_phi_4_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 9216), torch.bfloat16)],
        {
            "model_names": ["pt_alexnet_base_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Identity0,
        [((1, 100, 8, 25, 34), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Identity0,
        [((1, 384, 768), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 384, 384), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 12, 522, 522), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 20, 256, 256), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 24, 256, 256), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (Identity0, [((256, 1024), torch.float32)], {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 8, 256, 2048), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 256, 256), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 8, 2048, 256), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 39, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 28, 29, 29), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 28, 39, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 1536), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 24, 1, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 24, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((2, 1, 6144), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 3136, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Identity0,
        [((1, 784, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Identity0,
        [((1, 196, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (Identity0, [((1, 61, 4096), torch.float32)], {"model_names": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99}),
    (
        Identity0,
        [((1, 1, 4096), torch.float32)],
        {
            "model_names": ["pt_t5_t5_large_text_gen_hf", "pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Identity0,
        [((1, 1500, 1024), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 1500, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 1500, 4096), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Identity0,
        [((1, 16, 1, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Identity")

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
