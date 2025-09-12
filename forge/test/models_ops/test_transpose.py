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
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-4, dim1=-1)
        return transpose_output_1


class Transpose3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-3, dim1=-1)
        return transpose_output_1


class Transpose4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-5, dim1=-3)
        return transpose_output_1


class Transpose5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-4, dim1=-3)
        return transpose_output_1


class Transpose6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-5, dim1=-4)
        return transpose_output_1


class Transpose7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-4, dim1=-2)
        return transpose_output_1


class Transpose8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-5, dim1=-2)
        return transpose_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Transpose0,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "onnx_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 64, 128), torch.float32)],
        {
            "model_names": ["onnx_albert_xxlarge_v1_mlm_hf", "onnx_albert_xxlarge_v2_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 128, 64), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 784), torch.float32)],
        {
            "model_names": ["onnx_autoencoder_linear_img_enc_github", "pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 128), torch.float32)],
        {
            "model_names": ["onnx_autoencoder_linear_img_enc_github", "pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 64), torch.float32)],
        {
            "model_names": ["onnx_autoencoder_linear_img_enc_github", "pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 12), torch.float32)],
        {
            "model_names": ["onnx_autoencoder_linear_img_enc_github", "pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 3), torch.float32)],
        {
            "model_names": ["onnx_autoencoder_linear_img_enc_github", "pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 12), torch.float32)],
        {
            "model_names": ["onnx_autoencoder_linear_img_enc_github", "pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 64), torch.float32)],
        {
            "model_names": ["onnx_autoencoder_linear_img_enc_github", "pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((784, 128), torch.float32)],
        {
            "model_names": ["onnx_autoencoder_linear_img_enc_github", "pt_autoencoder_linear_img_enc_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 2304), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 768), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_albert_squad2_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_stereo_medium_music_generation_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_gptneo_gpt_neo_125m_seq_cls_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_roberta_xlm_base_mlm_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_t5_t5_base_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_albert_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_gpt_gpt2_text_gen_hf",
                "pt_stereo_small_music_generation_hf",
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_stereo_large_music_generation_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_speecht5_tts_tts_text_to_speech_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 7, 12, 64), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 7, 64), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 7, 64), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 7), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 7, 64), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_hrnet_hrnetv2_w64_img_cls_osmr",
                "onnx_wideresnet_wide_resnet50_2_img_cls_timm",
                "onnx_hrnet_hrnetv2_w30_img_cls_osmr",
                "onnx_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "onnx_resnext_resnext50_32x4d_img_cls_osmr",
                "onnx_hrnet_hrnet_w18_small_v1_img_cls_osmr",
                "onnx_hrnet_hrnetv2_w44_img_cls_osmr",
                "onnx_resnext_resnext101_64x4d_img_cls_osmr",
                "onnx_resnext_resnext14_32x4d_img_cls_osmr",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_hrnet_hrnetv2_w18_img_cls_osmr",
                "onnx_resnet_50_img_cls_hf",
                "onnx_xception_xception71_tf_in1k_img_cls_timm",
                "onnx_hrnet_hrnetv2_w48_img_cls_osmr",
                "onnx_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_xception_xception65_img_cls_timm",
                "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr",
                "onnx_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "onnx_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 196), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 512), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 512), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_s16_224_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_mlp_mixer_mixer_s32_224_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1280), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 576), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "onnx_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_mlp_mixer_mixer_l16_224_img_cls_timm",
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "onnx_googlenet_googlenet_img_cls_torchvision",
                "onnx_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 1, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 256, 64), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 128, 128, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4096, 2, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 2, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1024, 5, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 5, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1024, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 32, 32, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 120, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 12, 3, 8, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 12, 1, 8, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 12, 8, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 12, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 12, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 15, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 12, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 1, 12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 120, 12, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_squad2_qa_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_imdb_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 14, 12, 64), torch.float32)],
        {
            "model_names": ["pt_albert_squad2_qa_hf", "pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 14, 64), torch.float32)],
        {
            "model_names": ["pt_albert_squad2_qa_hf", "pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 14, 64), torch.float32)],
        {
            "model_names": ["pt_albert_squad2_qa_hf", "pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 14, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((3072, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_squad2_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_stereo_medium_music_generation_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_gptneo_gpt_neo_125m_seq_cls_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_roberta_xlm_base_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_t5_t5_base_text_gen_hf",
                "pt_albert_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_speecht5_tts_tts_text_to_speech_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 3072), torch.float32)],
        {
            "model_names": [
                "pt_albert_squad2_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_stereo_medium_music_generation_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_gptneo_gpt_neo_125m_seq_cls_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_roberta_xlm_base_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_t5_t5_base_text_gen_hf",
                "pt_albert_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_speecht5_tts_tts_text_to_speech_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_qwen1_5_0_5b_clm_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_qwen1_5_0_5b_chat_clm_hf",
                "pt_llava_1_5_7b_cond_gen_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_xglm_xglm_564m_clm_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "onnx_albert_large_v1_mlm_hf",
                "onnx_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "onnx_albert_large_v1_mlm_hf",
                "onnx_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "onnx_albert_large_v1_mlm_hf",
                "onnx_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_names": ["onnx_albert_large_v1_mlm_hf", "onnx_albert_large_v2_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_llava_1_5_7b_cond_gen_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_xglm_xglm_564m_clm_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 4096), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_llava_1_5_7b_cond_gen_hf",
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_xglm_xglm_564m_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((9, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "onnx_mobilenetv1_mobilenet_v1_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 3072), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_xglm_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2208), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_vit_vit_h_14_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1792), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2560), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_64x4d_osmr_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_img_cls_timm",
                "pt_hrnet_hrnet_w30_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w32_img_cls_timm",
                "pt_hrnet_hrnet_w40_img_cls_timm",
                "pt_hrnet_hrnet_w44_img_cls_timm",
                "pt_hrnet_hrnet_w48_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                "pt_resnet_resnet50_timm_img_cls_timm",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_osmr_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_timm_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_timm_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_osmr_img_cls_osmr",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_hrnet_hrnet_w64_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_xception_xception65_img_cls_timm",
                "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose6,
        [((1, 3, 16, 16, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-4"},
        },
    ),
    (
        Transpose7,
        [((1, 16, 3, 16, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 16, 16, 3, 16), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_yolo_world_default_obj_det_github",
                "pt_detr_resnet_50_obj_det_hf",
                "pt_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_base_img_cls_hf",
                "pt_beit_base_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((196, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_vilt_vqa_qa_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_base_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vilt_mlm_mlm_hf",
                "pt_vit_vit_b_32_img_cls_torchvision",
                "pt_beit_base_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mgp_default_scene_text_recognition_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 3072), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_vilt_vqa_qa_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_base_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vilt_mlm_mlm_hf",
                "pt_vit_vit_b_32_img_cls_torchvision",
                "pt_beit_base_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mgp_default_scene_text_recognition_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((11221, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((196, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 49), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 49, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 49), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((49, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_gptneo_gpt_neo_125m_seq_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_albert_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 512), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 16, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 1024), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 768), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2048, 8, 32), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 32), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 2048, 32), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 768), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2048, 8, 160), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 2048, 160), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 160, 2048), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 160), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 160), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 1280), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 160), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 160, 256), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 96), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 96), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 96, 256), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 2048, 96), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((262, 768), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 2048), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_deepseek_1_3b_instruct_qa_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_seq_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_xglm_xglm_1_7b_clm_hf",
                "pt_llama3_llama_3_2_1b_seq_cls_hf",
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_qwen_v3_1_7b_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_stereo_large_music_generation_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
                "pt_gemma_google_gemma_1_1_2b_it_qa_hf",
                "pt_gemma_google_gemma_2b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8192, 2048), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_seq_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_xglm_xglm_1_7b_clm_hf",
                "pt_llama3_llama_3_2_1b_seq_cls_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_stereo_large_music_generation_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 8192), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_seq_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_xglm_xglm_1_7b_clm_hf",
                "pt_llama3_llama_3_2_1b_seq_cls_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_stereo_large_music_generation_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51200, 2048), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf", "pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((896, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_v2_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 14, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 32, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((128, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_v2_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 2, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((14, 35, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 14, 35, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((4864, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_v2_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((896, 4864), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_v2_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((151936, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_v2_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 1, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 128, 128, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4096, 2, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 2, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_yolo_world_default_obj_det_github",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 64, 64, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((320, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1024, 5, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 5, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1024, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1280, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
                "pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((320, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
                "pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 32, 32, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 1536), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_qwen_v2_1_5b_clm_hf",
                "pt_bloom_default_clm_hf",
                "pt_qwen_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 24, 64), torch.float32)],
        {"model_names": ["pt_stereo_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((24, 1, 64), torch.float32)],
        {"model_names": ["pt_stereo_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 24, 1, 64), torch.float32)],
        {"model_names": ["pt_stereo_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose0,
        [((1, 25, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 25, 64), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((25, 25, 12), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 25, 25), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 25, 64), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 25, 64), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 25), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 768), torch.float32)],
        {
            "model_names": ["pt_stereo_medium_music_generation_hf", "onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 25, 24, 64), torch.float32)],
        {"model_names": ["pt_stereo_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((24, 25, 64), torch.float32)],
        {"model_names": ["pt_stereo_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((6144, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_medium_music_generation_hf", "pt_bloom_default_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 6144), torch.float32)],
        {
            "model_names": ["pt_stereo_medium_music_generation_hf", "pt_bloom_default_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 1536), torch.float32)],
        {"model_names": ["pt_stereo_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 56, 128, 56), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 8, 7, 8, 7, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((384, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((64, 49, 3, 4, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 64, 4, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 64, 49, 4, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((49, 49, 4), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 8, 8, 7, 7, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_yolo_world_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 7, 4, 7, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((768, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((16, 49, 3, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 16, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 16, 49, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((49, 49, 8), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 4, 7, 7, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 7, 2, 7, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((1536, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((4, 49, 3, 16, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 4, 16, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 4, 49, 16, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((49, 49, 16), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 2, 7, 7, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((1024, 2048), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision", "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 7, 1, 7, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((3072, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 49, 3, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 1, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 49, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((49, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_beit_large_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_vit_large_img_cls_hf",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 1, 7, 7, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((4096, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_beit_large_img_cls_hf",
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vit_large_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_beit_large_img_cls_hf",
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vit_large_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 7, 7, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_beit_large_img_cls_hf",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_vit_large_img_cls_hf",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 25088), torch.float32)],
        {
            "model_names": [
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 4096), torch.float32)],
        {
            "model_names": [
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "onnx_alexnet_base_img_cls_torchhub",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
                "pt_llava_1_5_7b_cond_gen_hf",
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
                "pt_deepseek_7b_instruct_qa_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 4096), torch.float32)],
        {
            "model_names": [
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "onnx_alexnet_base_img_cls_torchhub",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 25088), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_torchvision_vgg16_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_torchvision_vgg19_img_cls_torchvision",
                "pt_vgg_torchvision_vgg13_bn_img_cls_torchvision",
                "pt_vgg_torchvision_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_torchvision_vgg19_bn_img_cls_torchvision",
                "pt_vgg_torchvision_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_torchvision_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_torchvision_vgg11_img_cls_torchvision",
                "pt_vgg_bn_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_torchvision_vgg16_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_alexnet_base_img_cls_osmr",
                "pt_alexnet_base_img_cls_torchhub",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_torchvision_vgg19_img_cls_torchvision",
                "pt_vgg_torchvision_vgg13_bn_img_cls_torchvision",
                "pt_vgg_torchvision_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_torchvision_vgg19_bn_img_cls_torchvision",
                "pt_vgg_torchvision_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_vgg_torchvision_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_torchvision_vgg11_img_cls_torchvision",
                "pt_vgg_bn_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_torchvision_vgg16_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_alexnet_base_img_cls_osmr",
                "pt_alexnet_base_img_cls_torchhub",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_torchvision_vgg19_img_cls_torchvision",
                "pt_vgg_torchvision_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_torchvision_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_torchvision_vgg19_bn_img_cls_torchvision",
                "pt_vgg_torchvision_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_torchvision_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_torchvision_vgg11_img_cls_torchvision",
                "pt_vgg_bn_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 80, 85, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 40, 85, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 20, 85, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 16, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 85, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2016), torch.float32)],
        {
            "model_names": ["regnet_regnety_080_onnx", "onnx_regnet_facebook_regnet_y_080_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1536), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 13, 12, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 13, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 13, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 32, 13), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 13, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 384), torch.float32)],
        {
            "model_names": [
                "onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 196), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 768), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_b16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 768), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_b16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 9216), torch.float32)],
        {"model_names": ["onnx_mnist_base_img_cls_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((10, 128), torch.float32)],
        {"model_names": ["onnx_mnist_base_img_cls_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 384, 1500), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 6, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 1500), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 1500, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 32, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2, 32, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2, 400, 32), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 2, 400, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2, 400, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 4, 16, 8400), torch.float32)],
        {
            "model_names": [
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
                "onnx_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 14), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((1, 14, 2), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2, 14, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pt_albert_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pt_albert_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 9), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pt_albert_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 9, 2), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_uncased_qa_padlenlp", "pd_ernie_1_0_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 9, 1), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_uncased_qa_padlenlp", "pd_ernie_1_0_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 197), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 197, 64), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 288, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 25, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((25, 2, 1, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 768), torch.float32)],
        {
            "model_names": ["pt_albert_base_v1_mlm_hf", "pt_albert_base_v2_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((30000, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 9216), torch.bfloat16)],
        {
            "model_names": [
                "pt_alexnet_base_img_cls_osmr",
                "pt_alexnet_base_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_large_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vit_large_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf", "pt_vit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_large_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vit_large_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 27, 27, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 27, 16, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((197, 197, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf", "pt_vit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose7,
        [((1, 16, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 588, 16, 128), torch.float32)],
        {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 588), torch.float32)],
        {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 588, 128), torch.float32)],
        {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 588, 128), torch.float32)],
        {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((5504, 2048), torch.float32)],
        {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2048, 5504), torch.float32)],
        {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32256, 2048), torch.float32)],
        {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 1920), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 49), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm", "pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 49, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 49), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((49, 384), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_base_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vit_vit_b_32_img_cls_torchvision",
                "pt_beit_base_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 960), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 72), torch.float32)],
        {
            "model_names": ["pt_nbeats_generic_basis_time_series_forecasting_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 512), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_generic_basis_time_series_forecasting_github",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "pt_t5_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 512), torch.float32)],
        {
            "model_names": ["pt_nbeats_generic_basis_time_series_forecasting_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 12), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 2048), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_gptneo_gpt_neo_1_3b_seq_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_llama3_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1088), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 768), torch.float32)],
        {
            "model_names": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 16, 16, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 320), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 96, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 56, 96, 56), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 8, 7, 8, 7, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((288, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((64, 49, 3, 3, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 64, 3, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 64, 49, 3, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((49, 49, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 49, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 32, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 8, 8, 7, 7, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((384, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 7, 4, 7, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((576, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((16, 49, 3, 6, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 16, 6, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 16, 49, 6, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((96, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((49, 49, 6), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 49, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((96, 32, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_deit_tiny_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 4, 7, 7, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((768, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_deit_tiny_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_deit_tiny_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 7, 2, 7, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((1152, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((4, 49, 3, 12, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 4, 12, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 4, 49, 12, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((49, 49, 12), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 49, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 32, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_deit_small_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 2, 7, 7, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((1536, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_deit_small_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 1536), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_deit_small_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 1536), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 7, 1, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((2304, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vit_vit_b_32_img_cls_torchvision",
                "pt_mgp_default_scene_text_recognition_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 49, 3, 24, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 1, 24, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 49, 24, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((49, 49, 24), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((24, 49, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 32, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_vilt_vqa_qa_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_base_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vilt_mlm_mlm_hf",
                "pt_vit_vit_b_32_img_cls_torchvision",
                "pt_beit_base_img_cls_hf",
                "pt_mgp_default_scene_text_recognition_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose3,
        [((1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 10, 85, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 85, 8400), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 85, 3549), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 3024), torch.float32)],
        {
            "model_names": ["regnet_regnety_160_onnx", "onnx_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_qwen_v3_1_7b_clm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_albert_xlarge_v2_mlm_hf", "onnx_albert_xlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_qwen_v3_1_7b_clm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_qwen_v3_1_7b_clm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 128), torch.float32)],
        {
            "model_names": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 256), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 257, 3, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 257, 1, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 257, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 257), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 257, 64), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 257, 768), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 257), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 49), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 49, 1024), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 2048, 32), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 2048), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 2048, 160), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 160), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 96), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 1, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 128, 128, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 128, 128), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4096, 2, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 2, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4096, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 64, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1024, 5, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 5, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1024, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 640), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 640, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 32, 32, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 16, 16, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 8, 64), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "onnx_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "onnx_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "onnx_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "onnx_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "onnx_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 1, 8), torch.float32)],
        {"model_names": ["onnx_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 1, 1), torch.float32)],
        {"model_names": ["onnx_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 61, 8, 64), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 61, 64), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 61, 64), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 61), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 61, 64), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((61, 61, 8), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 61, 61), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 1500), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 8, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 1500), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 1500, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16384, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 16384), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 4096), torch.float32)],
        {
            "model_names": ["pt_albert_xxlarge_v2_mlm_hf", "pt_albert_xxlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 5, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 5, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 5, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 5), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51200, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((9, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 384, 12, 64), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 384, 64), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 384, 64), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1000, 1536), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1001, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 256, 64), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((50272, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((261, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 224, 3, 224), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((261, 261), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 50176, 261), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 50176, 1, 261), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 50176, 261), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 261, 50176), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 261), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 512, 8, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 128, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 512, 1, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 29, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 29, 2, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8960, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 8960), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((151936, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 8, 8, 8, 8, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((384, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((64, 64, 3, 4, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose5,
        [((3, 64, 64, 4, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 64, 64, 4, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((512, 2), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((64, 64, 4), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((4, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((64, 4, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 4, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((128, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((512, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((128, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 4, 8, 4, 8, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((768, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision", "pt_speecht5_tts_tts_text_to_speech_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((16, 64, 3, 8, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose5,
        [((3, 64, 16, 8, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 16, 64, 8, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((64, 64, 8), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 8, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((16, 8, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256, 256), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_nbeats_trend_basis_time_series_forecasting_github",
                "pt_speecht5_tts_tts_text_to_speech_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 4, 8, 8, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((1024, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 2, 8, 2, 8, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((1536, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((4, 64, 3, 16, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose5,
        [((3, 64, 4, 16, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 4, 64, 16, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((64, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((64, 64, 16), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((4, 16, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((4, 16, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 2, 2, 8, 8, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((2048, 512), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "pt_t5_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 2048), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_llama3_llama_3_2_1b_seq_cls_hf",
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 2048), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_qwen_v3_1_7b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 8, 1, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision", "pt_qwen_v3_0_6b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 64, 3, 32, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose5,
        [((3, 64, 1, 32, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 1, 64, 32, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((32, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 1, 8, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose3,
        [((1, 8, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 1024, 8, 8), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 3, 85, 60, 60), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 60, 85, 60), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 30, 30), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 30, 85, 30), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 15, 15), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 15, 85, 15), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4, 5880), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 80, 5880), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1296), torch.float32)],
        {
            "model_names": ["regnet_regnety_064_onnx", "onnx_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 128), torch.float32)],
        {
            "model_names": ["onnx_albert_large_v1_mlm_hf", "onnx_albert_large_v2_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 9216), torch.float32)],
        {"model_names": ["onnx_alexnet_base_img_cls_torchhub"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 1664), torch.float32)],
        {
            "model_names": ["onnx_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 49), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_b32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 49, 768), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_b32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 64, 55, 55), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 55, 64, 55), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3025, 1, 322), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 3025, 322), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 322, 3025), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3025, 322), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 512, 8, 128), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 512, 128), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 512, 128), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 128, 512), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 512, 128), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 512, 1, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 512, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 512, 1024), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 512), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_names": [
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 128), torch.float32)],
        {
            "model_names": [
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 201, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 201, 64), torch.float32)],
        {
            "model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 201, 64), torch.float32)],
        {
            "model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 201), torch.float32)],
        {
            "model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 201, 64), torch.float32)],
        {
            "model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3129, 1536), torch.float32)],
        {
            "model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 40, 40), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 40, 85, 40), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 20, 20), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 20, 85, 20), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 10, 10), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 10, 85, 10), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 11, 12, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 26, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((21128, 128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 15, 12, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 15, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 15, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 15), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 15, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 8, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 8, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 8), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 8, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 2048), torch.float32)],
        {
            "model_names": ["pt_albert_xlarge_v1_mlm_hf", "pt_albert_xlarge_v2_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 522, 8, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 522), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 522, 4, 256), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((131072, 2048), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 16, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 5, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 16, 5, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 5, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 128, 5), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 1408), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1792), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 196), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_l16_224_img_cls_timm",
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_l16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 96, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((64, 64, 3, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 3, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 3, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 32, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((64, 64, 3), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 8, 4, 8, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((16, 64, 6, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 6, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((16, 6, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((96, 32, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((64, 64, 6), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 4, 8, 8, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 8, 2, 8, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((4, 64, 12, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((4, 12, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 12, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 32, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((48, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((64, 64, 12), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 2, 8, 8, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 8, 1, 8, 768), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((1, 64, 24, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 24, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 32, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((24, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((64, 64, 24), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((24, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 3, 32, 32, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 8, 32, 32, 8), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 8, 32, 32, 8), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 576), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 577, 3, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 577, 1, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 577, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 577, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 577, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 577), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 577, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 5, 12, 64), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((4, 12, 5, 64), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 12, 5, 64), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 64, 5), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((48, 5, 64), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4, 1), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((18000, 768), torch.float32)],
        {"model_names": ["pd_ernie_1_0_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 11, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 11, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 11, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 11), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 11, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((21128, 768), torch.float32)],
        {
            "model_names": ["pd_roberta_rbt4_ch_clm_padlenlp", "pd_bert_chinese_roberta_base_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((30522, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 192, 196), torch.bfloat16)],
        {
            "model_names": ["pt_deit_tiny_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 3, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_tiny_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_tiny_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_tiny_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1000, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_tiny_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 19200), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 19200, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 1, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 19200, 256), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 19200), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 120, 160, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 160, 120), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 4800), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4800, 2, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4800, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 2, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4800, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4800, 512), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 4800), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 60, 80, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 80, 60), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 1200), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1200, 5, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1200, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 5, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1200, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1200, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1200), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 30, 40, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 40, 30), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 8, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 300, 2048), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 15, 20, 512), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 20, 15), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 49), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm", "pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 49, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 49), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((49, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 9216), torch.bfloat16)],
        {
            "model_names": ["pt_mnist_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((10, 128), torch.bfloat16)],
        {
            "model_names": ["pt_mnist_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 16, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 32, 6), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_0_5b_clm_hf", "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((2816, 1024), torch.float32)],
        {
            "model_names": [
                "pt_qwen1_5_0_5b_clm_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_qwen1_5_0_5b_chat_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 2816), torch.float32)],
        {
            "model_names": [
                "pt_qwen1_5_0_5b_clm_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_qwen1_5_0_5b_chat_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((151936, 1024), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_0_5b_clm_hf", "pt_qwen_v3_0_6b_clm_hf", "pt_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 2560), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 31, 32, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 31), torch.bfloat16)],
        {
            "model_names": [
                "pt_qwen_v3_embedding_4b_sentence_embed_gen_hf",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 2560), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 31, 8, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_qwen_v3_embedding_4b_sentence_embed_gen_hf",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 32, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2560, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((9728, 2560), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2560, 9728), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 912), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 3712), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_320_img_cls_hf", "pt_regnet_regnet_y_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1512), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 96, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 96, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((288, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((64, 64, 3, 3, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 64, 64, 3, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 64, 64, 3, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3, 512), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((576, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((16, 64, 3, 6, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 64, 16, 6, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 16, 64, 6, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 512), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 768), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 768), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1152, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((4, 64, 3, 12, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 64, 4, 12, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 4, 64, 12, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 512), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 1536), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 1536), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2304, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 64, 3, 24, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 64, 1, 24, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 64, 24, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 512), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 1, 8, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose3,
        [((1, 8, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 8, 8), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 513, 16, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 513, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((513, 513, 16), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 513, 513), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 513, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 513, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 513), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 61, 16, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((61, 61, 16), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 61, 61), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 61), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32128, 1024), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 12, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_deit_base_img_cls_hf",
                "pt_beit_base_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_base_img_cls_hf",
                "pt_beit_base_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_deit_base_img_cls_hf",
                "pt_beit_base_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose7,
        [((1, 12, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 85, 3549), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 384, 196), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 197), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 384), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((11221, 768), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 1024), torch.float32)],
        {
            "model_names": ["pt_albert_large_v2_mlm_hf", "pt_albert_large_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 1024), torch.float32)],
        {
            "model_names": ["pt_albert_large_v2_token_cls_hf", "pt_albert_large_v1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 12, 64), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 16, 64), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 16, 64), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 384, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 384, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 384, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 384, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4608, 1536), torch.float32)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 16, 96), torch.float32)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 32, 96), torch.float32)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 96, 32), torch.float32)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 32, 96), torch.float32)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((250880, 1536), torch.float32)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((28996, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((21843, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 12, 64), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 5, 64), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 12, 5, 64), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 5, 64), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 64, 5), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 196, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1001, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 1024, 72), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_generic_basis_time_series_forecasting_github",
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "pt_nbeats_trend_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((256, 72), torch.float32)],
        {
            "model_names": ["pt_nbeats_trend_basis_time_series_forecasting_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 256), torch.float32)],
        {
            "model_names": ["pt_nbeats_trend_basis_time_series_forecasting_github"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((50272, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 400), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2240), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_120_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 1, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 128, 128, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4096, 2, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 2, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4096, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 64, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1024, 5, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 5, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1024, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((640, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 640), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 640, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((160, 640), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 32, 32, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf", "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 12, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_speecht5_tts_tts_text_to_speech_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 1, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_speecht5_tts_tts_text_to_speech_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 1), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_speecht5_tts_tts_text_to_speech_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_speecht5_tts_tts_text_to_speech_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 201, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 201, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 201, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 201, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 201), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3129, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1369), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1370, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3840, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose8,
        [((1, 1370, 1, 3, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1370, 16, 80), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 1370, 80), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 16, 1370, 80), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1370, 16, 1, 80), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1280, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_vit_h_14_img_cls_torchvision",
                "pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1370, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5120, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_vit_h_14_img_cls_torchvision",
                "pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 5120), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_vit_h_14_img_cls_torchvision",
                "pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 16, 128), torch.float32)],
        {
            "model_names": ["pt_xglm_xglm_1_7b_clm_hf", "pt_gptneo_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 256, 128), torch.float32)],
        {
            "model_names": ["pt_xglm_xglm_1_7b_clm_hf", "pt_gptneo_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 256), torch.float32)],
        {
            "model_names": ["pt_xglm_xglm_1_7b_clm_hf", "pt_gptneo_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 256, 128), torch.float32)],
        {
            "model_names": ["pt_xglm_xglm_1_7b_clm_hf", "pt_gptneo_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 256, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256008, 2048), torch.float32)],
        {"model_names": ["pt_xglm_xglm_1_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 3, 224, 224), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 224, 3, 224), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((7, 7, 3, 64), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 7, 3, 7), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 3, 7, 7), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 64, 64), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 1, 64, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 64, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((3, 3, 64, 64), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 3, 64, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 64, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 64, 256), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((256, 1, 64, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 64, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 1, 256, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 256, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 256, 128), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((128, 1, 256, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128, 256, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((3, 3, 128, 128), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((128, 3, 128, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128, 128, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 128, 512), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((512, 1, 128, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 128, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 256, 512), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((512, 1, 256, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 256, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 512, 128), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((128, 1, 512, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128, 512, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 512, 256), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((256, 1, 512, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 512, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((3, 3, 256, 256), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((256, 3, 256, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 256, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 256, 1024), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1024, 1, 256, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1024, 256, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1024, 1, 512, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1024, 512, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 1024, 256), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((256, 1, 1024, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 1024, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 1024, 512), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((512, 1, 1024, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 1024, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((3, 3, 512, 512), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((512, 3, 512, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 512, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 512, 2048), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    pytest.param(
        (
            Transpose0,
            [((2048, 1, 512, 1), torch.float32)],
            {
                "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
                "pcc": 0.99,
                "args": {"dim0": "-3", "dim1": "-2"},
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    pytest.param(
        (
            Transpose1,
            [((2048, 512, 1, 1), torch.float32)],
            {
                "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
                "pcc": 0.99,
                "args": {"dim0": "-2", "dim1": "-1"},
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    (
        Transpose2,
        [((1, 1, 1024, 2048), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    pytest.param(
        (
            Transpose0,
            [((2048, 1, 1024, 1), torch.float32)],
            {
                "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
                "pcc": 0.99,
                "args": {"dim0": "-3", "dim1": "-2"},
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    pytest.param(
        (
            Transpose1,
            [((2048, 1024, 1, 1), torch.float32)],
            {
                "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
                "pcc": 0.99,
                "args": {"dim0": "-2", "dim1": "-1"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:141: tt::exception"
            )
        ],
    ),
    (
        Transpose2,
        [((1, 1, 2048, 512), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-1"},
        },
    ),
    pytest.param(
        (
            Transpose0,
            [((512, 1, 2048, 1), torch.float32)],
            {
                "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
                "pcc": 0.99,
                "args": {"dim0": "-3", "dim1": "-2"},
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    pytest.param(
        (
            Transpose1,
            [((512, 2048, 1, 1), torch.float32)],
            {
                "model_names": ["jax_resnet_50_img_cls_hf", "tf_resnet_resnet50_img_cls_keras"],
                "pcc": 0.99,
                "args": {"dim0": "-2", "dim1": "-1"},
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    (
        Transpose3,
        [((1, 1, 1, 2048), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 2048, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 27, 27), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 27, 12, 27), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((197, 197, 12), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 197, 197), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1408), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 256), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((50257, 768), torch.float32)],
        {
            "model_names": ["pt_gptneo_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 32, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 4), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 8, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((9, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_basic_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 72), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((48, 2048), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2560, 2560), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 11, 32, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 11), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((10240, 2560), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2560, 10240), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 2560), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_gptneo_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 1024), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_stereo_small_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf", "pt_qwen_v3_4b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 8, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf", "pt_qwen_v3_4b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1000, 1296), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 440), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 10, 12, 64), torch.float32)],
        {
            "model_names": ["pt_roberta_xlm_base_mlm_hf", "pd_bert_bert_base_japanese_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 10, 64), torch.float32)],
        {
            "model_names": ["pt_roberta_xlm_base_mlm_hf", "pd_bert_bert_base_japanese_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 10, 64), torch.float32)],
        {
            "model_names": ["pt_roberta_xlm_base_mlm_hf", "pd_bert_bert_base_japanese_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((250002, 768), torch.float32)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((768, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose8,
        [((1, 197, 1, 3, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 12, 1, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 50, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose8,
        [((1, 50, 1, 3, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((50, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 16, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((50, 16, 1, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 32, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 400, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 400, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 3712), torch.float32)],
        {
            "model_names": ["regnet_regnety_320_onnx", "onnx_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 16, 64), torch.float32)],
        {
            "model_names": [
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 197, 64), torch.float32)],
        {
            "model_names": [
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 197, 64), torch.float32)],
        {
            "model_names": [
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 197), torch.float32)],
        {
            "model_names": [
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 197, 64), torch.float32)],
        {
            "model_names": [
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 27, 27), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 27, 16, 27), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((197, 197, 16), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 197, 197), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 192, 196), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 3, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3, 64, 197), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 192), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2208), torch.float32)],
        {
            "model_names": ["onnx_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 7, 8, 64), torch.float32)],
        {
            "model_names": [
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 7, 64), torch.float32)],
        {
            "model_names": [
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 8, 7, 64), torch.float32)],
        {
            "model_names": [
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 8, 7, 64), torch.float32)],
        {
            "model_names": ["onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((50257, 2048), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1001, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 224, 256, 224), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 50176, 512), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 50176, 1, 512), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 50176, 512), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 50176), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 512), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 32, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 5), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 5, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 5, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6144, 2048), torch.float32)],
        {"model_names": ["pt_qwen_v3_1_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2048, 6144), torch.float32)],
        {"model_names": ["pt_qwen_v3_1_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((151936, 2048), torch.float32)],
        {"model_names": ["pt_qwen_v3_1_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 2016), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 96, 3136), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 49, 3, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((16, 49, 6, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((4, 49, 12, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 49, 24, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 513, 12, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 513, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((513, 513, 12), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 513, 513), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 513, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 513, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 513), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 61, 12, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((61, 61, 12), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 61, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32128, 768), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 204, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 204, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 204, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 204, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 204), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 50, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose8,
        [((1, 50, 1, 3, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((50, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 12, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((50, 12, 1, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((50, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 160, 85, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 17, 4480), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 17, 1120), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 17, 280), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((21843, 768), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 49), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 49, 512), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 1500), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 12, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 1500), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 1500, 64), torch.float32)],
        {
            "model_names": [
                "onnx_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32000, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((1, 27, 27, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_base_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_base_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_base_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 27, 12, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_base_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((197, 197, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_base_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_base_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1664), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((119547, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_multilingual_cased_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((21843, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((50272, 512), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 35, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 35, 2, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1000, 888), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 16, 16, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 16, 64), torch.float32)],
        {"model_names": ["pt_stereo_small_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 1, 64), torch.float32)],
        {"model_names": ["pt_stereo_small_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 1, 64), torch.float32)],
        {"model_names": ["pt_stereo_small_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1024, 768), torch.float32)],
        {"model_names": ["pt_stereo_small_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 25, 16, 64), torch.float32)],
        {"model_names": ["pt_stereo_small_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 25, 64), torch.float32)],
        {"model_names": ["pt_stereo_small_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 513, 8, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((513, 513, 8), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32128, 512), torch.float32)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose8,
        [((1, 197, 1, 3, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 16, 1, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 80, 4, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4, 80, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 512), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 80, 2, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 80, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 512, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 80, 512, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 80, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 27), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 27, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 27, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 80, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 80, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 256), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 40, 512, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 20, 512, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 32, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 400, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 400, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2240), torch.float32)],
        {
            "model_names": ["regnet_regnety_120_onnx", "onnx_regnet_facebook_regnet_y_120_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 7), torch.float32)],
        {
            "model_names": ["onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 100, 8, 32), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 100, 32), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 100), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 100, 32), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 280), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 280, 8, 32), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 280, 32), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 280), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 280, 32), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 224, 256, 224), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 50176, 1, 512), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 50176, 512), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 50176), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 50176, 512), torch.float32)],
        {
            "model_names": ["onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 11, 2), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 11, 1), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 4, 12, 64), torch.float32)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 12, 4, 64), torch.float32)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 12, 4, 64), torch.float32)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 64, 4), torch.float32)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((24, 4, 64), torch.float32)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((18, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet121_xray_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 257, 3, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 257, 1, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 257, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 257), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 257, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 257), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((38, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((50257, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((30522, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 576), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((322, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 64, 55, 55), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 55, 64, 55), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((322, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3025, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3025, 1, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 3025, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 322, 3025), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 29, 16, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 32, 29), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose0,
        [((1, 39, 14, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 32, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 39, 2, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((14, 39, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 14, 39, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((2, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1008), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 3024), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_16gf_img_cls_torchvision", "pt_regnet_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 784), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 512), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 513, 6, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((6, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((513, 513, 6), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((6, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 6, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 6, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((6, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((512, 384), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 61, 6, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((6, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose3,
        [((61, 61, 6), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((6, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 6, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 6, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((6, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 1088), torch.float32)],
        {
            "model_names": ["regnet_regnety_040_onnx", "onnx_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 334, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 334, 64), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 64, 334, 64), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 64, 334), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 334, 64), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 19200), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 19200, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 300), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 300, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 300, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 19200, 256), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 19200), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 120, 160, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 160, 120), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 4800), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4800, 2, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4800, 128), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 300), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 2, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 300, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 64, 300), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 300, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4800, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4800, 512), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 4800), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 60, 80, 128), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 80, 60), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 1200), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1200, 5, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1200, 320), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 300), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 5, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 300, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 64, 300), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 300, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1200, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1200, 1280), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1200), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 30, 40, 320), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 40, 30), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 300), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 300, 8, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 300, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 300, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 300), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 300, 64), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 300, 2048), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048, 300), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 15, 20, 512), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 20, 15), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12288, 4096), torch.float32)],
        {
            "model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf", "pt_ministral_ministral_8b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 334), torch.float32)],
        {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 1024, 576), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 577, 16, 64), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 577, 64), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 577, 64), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose0,
        [((1, 596, 32, 128), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 596), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 596, 128), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 596, 128), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((11008, 4096), torch.float32)],
        {
            "model_names": [
                "pt_llava_1_5_7b_cond_gen_hf",
                "pt_deepseek_7b_instruct_qa_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_huggyllama_7b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 11008), torch.float32)],
        {
            "model_names": [
                "pt_llava_1_5_7b_cond_gen_hf",
                "pt_deepseek_7b_instruct_qa_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_huggyllama_7b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32064, 4096), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 32, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 12), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
                "pt_phi4_microsoft_phi_4_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 8, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 12, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 12, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((14336, 4096), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 14336), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32000, 4096), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 12288), torch.float32)],
        {
            "model_names": ["pt_ministral_ministral_8b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((131072, 4096), torch.float32)],
        {
            "model_names": ["pt_ministral_ministral_8b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32768, 4096), torch.float32)],
        {"model_names": ["pt_mistral_7b_instruct_v03_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((9216, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 32, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 48, 256), torch.float32)],
        {"model_names": ["pt_phi3_5_mini_instruct_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 256, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 256, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8192, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 8192), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32064, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 7392), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 32, 64), torch.float32)],
        {"model_names": ["pt_stereo_large_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((32, 1, 64), torch.float32)],
        {"model_names": ["pt_stereo_large_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 1, 64), torch.float32)],
        {"model_names": ["pt_stereo_large_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((2048, 768), torch.float32)],
        {
            "model_names": ["pt_stereo_large_music_generation_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 25, 32, 64), torch.float32)],
        {"model_names": ["pt_stereo_large_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((32, 25, 64), torch.float32)],
        {"model_names": ["pt_stereo_large_music_generation_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 384, 196), torch.bfloat16)],
        {
            "model_names": ["pt_deit_small_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 6, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_small_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_small_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_small_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1000, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_small_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 512), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 2520), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 672), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 2816), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((640, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 640, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 640, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 64, 640, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((640, 640), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 4096, 10, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 4096, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 10, 4096, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((640, 2048), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 77, 10, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 77, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2560, 640), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((640, 2560), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((2, 64, 64, 640), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 1280, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 1280, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 32, 1280, 32), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 1024, 20, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((40, 1024, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 20, 1024, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1280, 2048), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 77, 20, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((40, 77, 64), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((2, 32, 32, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 20, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 2, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 2, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1500), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 20, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 1500, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 1500, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5120, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 5120), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256008, 1024), torch.float32)],
        {"model_names": ["pt_xglm_xglm_564m_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 100, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 100, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 100), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 100, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 25, 34, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 34, 25), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 850, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 850, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 850, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2048, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 2048), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((92, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((251, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 850, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128256, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_clm_hf", "pt_llama3_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 4, 4, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 38, 4, 4, 38), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 6, 4, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 19, 4, 6, 19), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 6, 4, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 10, 4, 6, 10), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 6, 4, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 5, 4, 6, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 4, 4, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 3, 4, 4, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 4, 4, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 1, 4, 4, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 4, 91, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 38, 91, 4, 38), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 6, 91, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 19, 91, 6, 19), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 6, 91, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 10, 91, 6, 10), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 6, 91, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 5, 91, 6, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 4, 91, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 3, 91, 4, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 4, 91, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 1, 91, 4, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51866, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 384), torch.float32)],
        {
            "model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 2), torch.float32)],
        {
            "model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1063, 32, 128), torch.float32)],
        {"model_names": ["pt_deepseek_7b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 1063), torch.float32)],
        {"model_names": ["pt_deepseek_7b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 1063, 128), torch.float32)],
        {"model_names": ["pt_deepseek_7b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 1063, 128), torch.float32)],
        {"model_names": ["pt_deepseek_7b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((102400, 4096), torch.float32)],
        {"model_names": ["pt_deepseek_7b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 522, 12, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3072, 9216), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((131072, 3072), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((23040, 3072), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 23040), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((18176, 4544), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4544, 18176), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4672, 4544), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 71, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((4544, 4544), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((65024, 4544), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 356, 8, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 128, 356), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 2048), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 356, 1, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 356, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 8, 356, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16384, 2048), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 16384), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256000, 2048), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 3072), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 356, 16, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 356, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 356, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((3072, 4096), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((24576, 3072), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((3072, 24576), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((256000, 3072), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 512, 8, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 128, 512), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 512, 1, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 512, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 8, 512, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose0,
        [((1, 256, 20, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((20, 256, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 20, 256, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 20, 256, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((20, 128, 256), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((50257, 2560), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 5, 20, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((20, 5, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 20, 5, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 20, 5, 128), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((20, 128, 5), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 256, 32, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 32, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 4), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128256, 4096), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 8, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 24, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_clm_hf", "pt_llama3_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_clm_hf", "pt_llama3_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_clm_hf", "pt_llama3_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128256, 3072), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 24, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_instruct_seq_cls_hf", "pt_llama3_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 4, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_instruct_seq_cls_hf", "pt_llama3_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 4, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_instruct_seq_cls_hf", "pt_llama3_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 3072), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 32, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((51200, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 32, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 12, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 12, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 96, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 32, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 48, 5), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 5, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 5, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 13, 32, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 48, 13), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 13, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 13, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((7680, 5120), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf", "pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 40, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 5), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 5, 10, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((40, 5, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 40, 5, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((5120, 5120), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf", "pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((17920, 5120), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf", "pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5120, 17920), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf", "pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 5120), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf", "pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 40, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose0,
        [((1, 12, 10, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((40, 12, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 40, 12, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((3584, 3584), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 13, 28, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 13), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((512, 3584), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 13, 4, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((28, 13, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 28, 13, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((18944, 3584), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((3584, 18944), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2, 3584), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((4096, 2560), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 128, 32, 128), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1024, 2560), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 128, 128), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2560, 4096), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((9728, 2560), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2560, 9728), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((151936, 2560), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2048, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 31, 16, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 16, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1024, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 2048), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 101, 8, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((51865, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 101, 20, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((51865, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 101, 16, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 16, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((51865, 1024), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 101, 12, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((51865, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 101, 6, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 101, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((51865, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 44, 24, 128), torch.float32)],
        {
            "model_names": ["pt_cogito_v1_preview_llama_3b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 44), torch.float32)],
        {
            "model_names": ["pt_cogito_v1_preview_llama_3b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 44, 8, 128), torch.float32)],
        {
            "model_names": ["pt_cogito_v1_preview_llama_3b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 44, 128), torch.float32)],
        {
            "model_names": ["pt_cogito_v1_preview_llama_3b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 44, 128), torch.float32)],
        {
            "model_names": ["pt_cogito_v1_preview_llama_3b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 80), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 24, 12, 64), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 24, 64), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((12, 24, 64), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((24, 24, 64), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((24, 12, 24), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 64, 24), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 24, 64), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((160, 768), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 2, 80), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 80, 2), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Transpose")

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
