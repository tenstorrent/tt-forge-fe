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
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Gelu0, [((2, 1, 8192), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Gelu0, [((2, 1, 4096), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Gelu0, [((2, 1, 6144), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Gelu0, [((1, 2, 3072), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Gelu0, [((1, 2, 5120), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Gelu0, [((1, 2, 4096), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Gelu0, [((1, 2, 1536), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Gelu0, [((1, 2, 2048), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Gelu0, [((1, 204, 3072), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Gelu0, [((1, 11, 768), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Gelu0, [((1, 201, 3072), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Gelu0, [((1, 1536), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (
        Gelu1,
        [((1, 128, 3072), torch.float32)],
        {"model_name": ["pt_albert_base_v2_masked_lm", "pt_albert_base_v2_token_cls"]},
    ),
    (
        Gelu0,
        [((1, 128, 3072), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
            ]
        },
    ),
    (
        Gelu1,
        [((1, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Gelu0,
        [((1, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
            ]
        },
    ),
    (
        Gelu0,
        [((1, 128, 8192), torch.float32)],
        {"model_name": ["pt_albert_xlarge_v1_token_cls", "pt_albert_xlarge_v1_masked_lm"]},
    ),
    (
        Gelu1,
        [((1, 128, 8192), torch.float32)],
        {"model_name": ["pt_albert_xlarge_v2_token_cls", "pt_albert_xlarge_v2_masked_lm"]},
    ),
    (
        Gelu0,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Gelu1,
        [((1, 128, 4096), torch.float32)],
        {"model_name": ["pt_albert_large_v2_token_cls", "pt_albert_large_v2_masked_lm"]},
    ),
    (
        Gelu0,
        [((1, 128, 16384), torch.float32)],
        {"model_name": ["pt_albert_xxlarge_v1_token_cls", "pt_albert_xxlarge_v1_masked_lm"]},
    ),
    (
        Gelu1,
        [((1, 128, 16384), torch.float32)],
        {"model_name": ["pt_albert_xxlarge_v2_masked_lm", "pt_albert_xxlarge_v2_token_cls"]},
    ),
    (Gelu0, [((1, 256, 4096), torch.float32)], {"model_name": ["pt_bart", "pt_xglm_564M"]}),
    (Gelu1, [((1, 256, 4096), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (
        Gelu0,
        [((1, 128, 768), torch.float32)],
        {"model_name": ["pt_bert_masked_lm", "pt_distilbert_masked_lm", "pt_roberta_masked_lm"]},
    ),
    (Gelu0, [((1, 384, 4096), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Gelu0, [((1, 384, 3072), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Gelu0, [((1, 6, 18176), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Gelu1, [((1, 7, 16384), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Gelu1, [((1, 256, 3072), torch.float32)], {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm"]}),
    (
        Gelu1,
        [((1, 256, 10240), torch.float32)],
        {"model_name": ["pt_gpt_neo_2_7B_causal_lm", "pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (Gelu1, [((1, 256, 8192), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm"]}),
    (Gelu0, [((1, 256, 8192), torch.float32)], {"model_name": ["pt_xglm_1_7B"]}),
    (Gelu1, [((1, 12, 10240), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Gelu1, [((1, 11, 10240), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Gelu0, [((1, 3072, 128), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Gelu1, [((1, 1, 2048), torch.float32)], {"model_name": ["pt_google_flan_t5_base"]}),
    (Gelu1, [((1, 1, 1024), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (
        Gelu0,
        [((1, 1, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (Gelu0, [((1, 197, 1536), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (
        Gelu0,
        [((1, 197, 3072), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (Gelu0, [((1, 197, 768), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (
        Gelu0,
        [((1, 1024, 512), torch.float32)],
        {"model_name": ["pt_mixer_l32_224", "pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]},
    ),
    (Gelu0, [((1, 49, 4096), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (Gelu0, [((1, 196, 4096), torch.float32)], {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]}),
    (Gelu0, [((1, 512, 256), torch.float32)], {"model_name": ["pt_mixer_s32_224", "pt_mixer_s16_224"]}),
    (Gelu0, [((1, 49, 2048), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (
        Gelu0,
        [((1, 768, 384), torch.float32)],
        {
            "model_name": [
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b32_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
            ]
        },
    ),
    (
        Gelu0,
        [((1, 196, 3072), torch.float32)],
        {
            "model_name": [
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
            ]
        },
    ),
    (Gelu0, [((1, 196, 2048), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Gelu0, [((1, 49, 3072), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (
        Gelu0,
        [((1, 512, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Gelu0,
        [((1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Gelu0,
        [((1, 4096, 512), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Gelu0,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Gelu0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (Gelu0, [((1, 16384, 128), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (Gelu0, [((1, 4096, 256), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (Gelu0, [((1, 1024, 640), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (Gelu0, [((1, 256, 1024), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (Gelu0, [((1, 4096, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Gelu0, [((1, 1024, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Gelu0, [((1, 256, 1536), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Gelu0, [((1, 64, 3072), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Gelu0, [((1, 197, 4096), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_property):
    record_property("frontend", "tt-forge-fe")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    for metadata_name, metadata_value in metadata.items():
        record_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
