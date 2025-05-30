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


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Transpose0,
        [((1, 6, 12, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 6), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 6, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
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
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
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
        Transpose1,
        [((1000, 1536), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 1, 256, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
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
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
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
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
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
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
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
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
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
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((30522, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_uncased_mlm_padlenlp", "pt_distilbert_distilbert_base_uncased_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 11, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
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
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
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
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
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
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
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
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
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
        Transpose1,
        [((18000, 768), torch.float32)],
        {"model_names": ["pd_ernie_1_0_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 120, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 12, 3, 8, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 12, 1, 8, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 12, 8, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 12, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 12, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 15, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 12, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 1, 12, 120), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 120, 12, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
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
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
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
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
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
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 768), torch.float32)],
        {"model_names": ["pt_albert_base_v2_mlm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((30000, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 128), torch.float32)],
        {
            "model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 128), torch.float32)],
        {
            "model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8192, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
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
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
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
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
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
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
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
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
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
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
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
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
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
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 4096), torch.float32)],
        {
            "model_names": ["pt_albert_xxlarge_v1_mlm_hf", "pt_albert_xxlarge_v2_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
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
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 256), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
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
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
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
        Transpose1,
        [((4096, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
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
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
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
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
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
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
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
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((9, 1024), torch.float32)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51200, 1024), torch.float32)],
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
        Transpose1,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 16, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 32, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 32, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 32, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 32), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 32, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf", "pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf", "pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf", "pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 64, 256), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf", "pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128256, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 12, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 64, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((2560, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 256, 32, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 16, 256), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((32, 80, 256), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((10240, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2560, 10240), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((51200, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 204, 12, 64), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 204, 64), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 12, 204, 64), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 204, 64), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 64, 204), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 1, 12, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 12, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
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
        [((1, 384, 16, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
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
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 384), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
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
        Transpose1,
        [((1, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
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
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
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
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1024), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 10, 12, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 64, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32000, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 8, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
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
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 9, 2), torch.float32)],
        {
            "model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 9, 1), torch.float32)],
        {
            "model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"],
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
        [((1024, 128), torch.float32)],
        {
            "model_names": ["pt_albert_large_v1_mlm_hf", "pt_albert_large_v2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 1024), torch.float32)],
        {"model_names": ["pt_albert_large_v1_mlm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2, 4096), torch.float32)],
        {"model_names": ["pt_albert_xxlarge_v1_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((4608, 1536), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 16, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 32, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 96, 32), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 32, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1536, 1536), torch.float32)],
        {
            "model_names": [
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6144, 1536), torch.float32)],
        {
            "model_names": [
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 6144), torch.float32)],
        {
            "model_names": [
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((250880, 1536), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((512, 512), torch.float32)],
        {
            "model_names": [
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 7, 8, 64), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 7, 64), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 7), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 8, 7, 64), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2048, 512), torch.float32)],
        {
            "model_names": [
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
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
        Transpose1,
        [((1, 12, 384, 64), torch.float32)],
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
        [((12, 64, 384), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 2304), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
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
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 7, 64), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 7, 64), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
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
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
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
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 32, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 4), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 8, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 64, 4), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
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
        Transpose1,
        [((1024, 512), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 16, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 64, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 512), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 7, 32, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 16, 7), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 7, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 32, 7, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 7, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((32, 64, 7), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((51200, 2048), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 201, 12, 64), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 201, 64), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 201, 64), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 201, 64), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 201), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 768), torch.float32)],
        {
            "model_names": [
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3129, 1536), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 384, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 6, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 1536), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
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
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 7, 3, 7), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 3, 7, 7), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 64, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 1, 64, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 64, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((3, 3, 64, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 3, 64, 3), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 64, 3, 3), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 64, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((256, 1, 64, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 64, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 1, 256, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 256, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 256, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((128, 1, 256, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 256, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((3, 3, 128, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((128, 3, 128, 3), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 128, 3, 3), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 128, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 1, 128, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 128, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 256, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 1, 256, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 256, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 512, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((128, 1, 512, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 512, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 512, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((256, 1, 512, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 512, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((3, 3, 256, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((256, 3, 256, 3), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 256, 3, 3), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 256, 1024), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1024, 1, 256, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1024, 256, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 512, 1024), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1024, 1, 512, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1024, 512, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 1024, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((256, 1, 1024, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 1024, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 1024, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 1, 1024, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 1024, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((3, 3, 512, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 3, 512, 3), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 512, 3, 3), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 512, 2048), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((2048, 1, 512, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((2048, 512, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 1024, 2048), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((2048, 1, 1024, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((2048, 1024, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose2,
        [((1, 1, 2048, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 1, 2048, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 2048, 1, 1), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
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
        Transpose1,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1792), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
        [((1, 768, 196), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 197), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 768), torch.float32)],
        {
            "model_names": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 16, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
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
        Transpose1,
        [((1, 288, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1280, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 20, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 2, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 20, 2, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 2, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 64, 2), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 20, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 20, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 20, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((20, 64, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5120, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1280, 5120), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1408), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2048), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
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
        [((1, 96, 4096), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((64, 64, 3, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 32, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((64, 64, 3), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 8, 4, 8, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((16, 64, 6, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((96, 32, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((64, 64, 6), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 4, 8, 8, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 8, 2, 8, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((4, 64, 12, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 32, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((48, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((64, 64, 12), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 2, 8, 8, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 8, 1, 8, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((1, 64, 24, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 32, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((24, 64, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((64, 64, 24), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((24, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 1024), torch.float32)],
        {"model_names": ["pt_albert_large_v2_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 14, 12, 64), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 14, 64), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 14, 64), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 14, 64), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 14), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 588, 16, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 588), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 588), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5504, 2048), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 5504), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32256, 2048), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
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
        Transpose1,
        [((28996, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 16, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 256, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 256, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 256, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 256), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((50257, 2048), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 4096), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 64), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 6, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((50280, 1024), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 1, 24, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 1, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((48, 64, 1), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 24, 1, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((2, 13, 12, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 13, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((13, 13, 12), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 13, 13), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 12, 13, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 12, 13, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 64, 13), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((2, 13, 24, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 13, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((48, 64, 13), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1, 8, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1500, 8, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
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
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Transpose")

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

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
