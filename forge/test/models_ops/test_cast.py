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
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.int32)
        return cast_output_1


class Cast2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.bool)
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
        [((1, 1, 1, 6), torch.int64)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 256, 256), torch.int64)],
        {
            "model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast2,
        [((1, 1, 256, 256), torch.bool)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast2,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast2,
        [((1, 1, 256, 256), torch.int32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast2,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 128), torch.bool)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast2,
        [((1, 128), torch.int32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 12, 128, 128), torch.bool)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 1, 32, 32), torch.bool)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast1,
        [((1, 1, 32, 32), torch.bool)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf", "pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 32, 32), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf", "pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf", "pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast2,
        [((1, 1, 32, 32), torch.int32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf", "pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 32, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 1, 8), torch.int64)],
        {
            "model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast3,
        [((1, 11), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.int64"}},
    ),
    (
        Cast1,
        [((1, 11), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast0,
        [((1, 11), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast2,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast2,
        [((1, 11), torch.int32)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast0,
        [((1, 1, 32), torch.int64)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((2, 1, 7, 7), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((2, 1, 7, 7), torch.bool)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 384), torch.int64)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 384), torch.bool)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast2,
        [((1, 384), torch.int32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 12, 384, 384), torch.bool)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 1, 7, 7), torch.bool)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 1, 7, 7), torch.bool)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast0,
        [((1, 1, 7, 7), torch.bool)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 1, 7), torch.int64)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 7), torch.bool)],
        {
            "model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast3,
        [((1,), torch.int32)],
        {
            "model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.int64"},
        },
    ),
    (
        Cast2,
        [((1, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast2,
        [((1, 1, 7, 7), torch.int32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast0,
        [((1, 1, 1, 15), torch.int64)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 14), torch.int64)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 6, 2048), torch.bool)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast0,
        [((2, 1, 1, 13), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((2, 13, 1), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((2, 1, 1, 13), torch.bool)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Cast")

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
