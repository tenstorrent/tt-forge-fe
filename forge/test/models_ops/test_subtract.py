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


class Subtract0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract0_const_0", shape=(1,), dtype=torch.float32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract0_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, subtract_input_0, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, subtract_input_1)
        return subtract_output_1


class Subtract2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract2_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract2_const_1"))
        return subtract_output_1


class Subtract3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract3_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract3_const_1"))
        return subtract_output_1


class Subtract4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract4_const_0", shape=(1,), dtype=torch.int32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract4_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract5_const_1", shape=(1,), dtype=torch.int32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract5_const_1"))
        return subtract_output_1


class Subtract6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract6_const_1", shape=(1, 256, 6, 20), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract6_const_1"))
        return subtract_output_1


class Subtract7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract7_const_1", shape=(1, 256, 12, 40), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract7_const_1"))
        return subtract_output_1


class Subtract8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract8_const_1", shape=(1, 128, 12, 40), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract8_const_1"))
        return subtract_output_1


class Subtract9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract9_const_1", shape=(1, 128, 24, 80), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract9_const_1"))
        return subtract_output_1


class Subtract10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract10_const_1", shape=(1, 64, 24, 80), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract10_const_1"))
        return subtract_output_1


class Subtract11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract11_const_1", shape=(1, 64, 48, 160), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract11_const_1"))
        return subtract_output_1


class Subtract12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract12_const_1", shape=(1, 32, 48, 160), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract12_const_1"))
        return subtract_output_1


class Subtract13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract13_const_1", shape=(1, 32, 96, 320), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract13_const_1"))
        return subtract_output_1


class Subtract14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract14_const_1", shape=(1, 16, 96, 320), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract14_const_1"))
        return subtract_output_1


class Subtract15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract15_const_1", shape=(1, 16, 192, 640), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract15_const_1"))
        return subtract_output_1


class Subtract16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract16_const_1", shape=(1, 256, 10, 32), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract16_const_1"))
        return subtract_output_1


class Subtract17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract17_const_1", shape=(1, 256, 20, 64), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract17_const_1"))
        return subtract_output_1


class Subtract18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract18_const_1", shape=(1, 128, 20, 64), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract18_const_1"))
        return subtract_output_1


class Subtract19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract19_const_1", shape=(1, 128, 40, 128), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract19_const_1"))
        return subtract_output_1


class Subtract20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract20_const_1", shape=(1, 64, 40, 128), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract20_const_1"))
        return subtract_output_1


class Subtract21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract21_const_1", shape=(1, 64, 80, 256), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract21_const_1"))
        return subtract_output_1


class Subtract22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract22_const_1", shape=(1, 32, 80, 256), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract22_const_1"))
        return subtract_output_1


class Subtract23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract23_const_1", shape=(1, 32, 160, 512), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract23_const_1"))
        return subtract_output_1


class Subtract24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract24_const_1", shape=(1, 16, 160, 512), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract24_const_1"))
        return subtract_output_1


class Subtract25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract25_const_1", shape=(1, 16, 320, 1024), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract25_const_1"))
        return subtract_output_1


class Subtract26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract26_const_0", shape=(5880, 2), dtype=torch.float32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract26_const_0"), subtract_input_1)
        return subtract_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Subtract0,
        [((1, 1, 1, 6), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract1,
        [((100, 8, 9240), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((100, 8, 4480), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((100, 8, 8640), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((100, 8, 17280), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((100, 8, 34240), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((2, 1, 1, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((2, 1, 7, 7), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Subtract2,
        [((2441216,), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 1, 1, 204), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 1, 1, 201), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 1, 1, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((1, 1, 1, 14), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 1, 1, 9), torch.float32)],
        {"model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Subtract3,
            [((1, 32), torch.int64)],
            {
                "model_name": [
                    "pt_bloom_bigscience_bloom_1b1_clm_hf",
                    "pt_opt_facebook_opt_1_3b_qa_hf",
                    "pt_opt_facebook_opt_350m_seq_cls_hf",
                    "pt_opt_facebook_opt_350m_qa_hf",
                    "pt_opt_facebook_opt_125m_qa_hf",
                    "pt_opt_facebook_opt_125m_seq_cls_hf",
                    "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Subtract4,
            [((1, 384), torch.int32)],
            {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    (
        Subtract0,
        [((1, 12, 384, 384), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Subtract4,
            [((1, 128), torch.int32)],
            {
                "model_name": [
                    "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                    "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                    "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                    "pt_distilbert_distilbert_base_cased_mlm_hf",
                    "pt_distilbert_distilbert_base_uncased_mlm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    (
        Subtract0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((1, 1, 1, 7), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract5,
        [((1,), torch.int32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Subtract0, [((1, 1, 1, 256), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    pytest.param(
        (
            Subtract4,
            [((1, 1, 256, 256), torch.int32)],
            {
                "model_name": [
                    "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                    "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                    "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                    "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                    "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                    "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                    "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                    "pt_phi2_microsoft_phi_2_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    pytest.param(
        (Subtract4, [((1, 4), torch.int32)], {"model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99}),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    pytest.param(
        (
            Subtract4,
            [((1, 1, 32, 32), torch.int32)],
            {
                "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf", "pt_llama3_huggyllama_llama_7b_clm_hf"],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    (
        Subtract0,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Subtract3,
            [((1, 256), torch.int64)],
            {
                "model_name": [
                    "pt_opt_facebook_opt_1_3b_clm_hf",
                    "pt_opt_facebook_opt_350m_clm_hf",
                    "pt_opt_facebook_opt_125m_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Subtract0,
        [((1, 1, 1, 2048), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1024, 72), torch.float32), ((1024, 72), torch.float32)],
        {
            "model_name": [
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract2,
        [((1, 3, 192, 640), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract6,
        [((1, 256, 6, 20), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract7,
        [((1, 256, 12, 40), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract8,
        [((1, 128, 12, 40), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract9,
        [((1, 128, 24, 80), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract10,
        [((1, 64, 24, 80), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract11,
        [((1, 64, 48, 160), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract12,
        [((1, 32, 48, 160), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract13,
        [((1, 32, 96, 320), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract14,
        [((1, 16, 96, 320), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract15,
        [((1, 16, 192, 640), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract2,
        [((1, 3, 320, 1024), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract16,
        [((1, 256, 10, 32), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract17,
        [((1, 256, 20, 64), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract18,
        [((1, 128, 20, 64), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract19,
        [((1, 128, 40, 128), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract20,
        [((1, 64, 40, 128), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract21,
        [((1, 64, 80, 256), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract22,
        [((1, 32, 80, 256), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract23,
        [((1, 32, 160, 512), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract24,
        [((1, 16, 160, 512), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract25,
        [((1, 16, 320, 1024), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract26,
        [((1, 5880, 2), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract1,
        [((1, 5880, 2), torch.float32), ((1, 5880, 2), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()
    forge_property_recorder.record_forge_op_name("Subtract")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_name":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "op_params":
            forge_property_recorder.record_forge_op_args(metadata_value)
        else:
            logger.warning("no utility function in forge property handler")

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

    forge_property_recorder.record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
