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
        [((2, 1, 1, 13), torch.int64)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((2, 13, 1), torch.int64)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((2, 1, 1, 13), torch.bool)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((2, 1, 7, 7), torch.int64)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((2, 1, 7, 7), torch.bool)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 596, 4096), torch.bool)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((2441216,), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast2,
        [((2441216,), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 9), torch.int64)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 14), torch.int64)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 256, 256), torch.int64)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast2,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 6), torch.int64)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 32), torch.int64)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "op_params": {"dtype": "torch.float32"}},
    ),
    (
        Cast1,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast2,
        [((1, 128), torch.bool)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast1,
        [((1, 128), torch.int32)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 128), torch.int32)],
        {
            "model_name": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int64"},
        },
    ),
    (
        Cast0,
        [((1, 12, 128, 128), torch.bool)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 384), torch.int64)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast2,
        [((1, 384), torch.bool)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast1,
        [((1, 384), torch.int32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 12, 384, 384), torch.bool)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 1, 7, 7), torch.bool)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 7), torch.int64)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 7), torch.bool)],
        {
            "model_name": ["pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast3,
        [((1,), torch.int32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int64"},
        },
    ),
    (
        Cast1,
        [((1, 1, 32, 32), torch.bool)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast2,
        [((1, 1, 32, 32), torch.bool)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf", "pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 32, 32), torch.bool)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 4), torch.int64)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99, "op_params": {"dtype": "torch.bool"}},
    ),
    (
        Cast2,
        [((1, 4), torch.bool)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast1,
        [((1, 4), torch.int32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99, "op_params": {"dtype": "torch.bool"}},
    ),
    (
        Cast1,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 1, 256, 256), torch.int32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf", "pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 1, 32, 32), torch.int32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf", "pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 32, 32), torch.int64)],
        {
            "model_name": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 32), torch.bool)],
        {
            "model_name": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 2048), torch.int64)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 16, 320, 1024), torch.bool)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 16, 192, 640), torch.bool)],
        {
            "model_name": [
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dtype": "torch.bool"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Cast")

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
