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


class Subtract0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract0_const_0", shape=(1,), dtype=torch.float32, use_random_value=True)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract0_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract1_const_0", shape=(1,), dtype=torch.int32, use_random_value=True)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract1_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract2_const_1", shape=(1,), dtype=torch.int64, use_random_value=True)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract2_const_1"))
        return subtract_output_1


class Subtract3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract3_const_1", shape=(1,), dtype=torch.int32, use_random_value=True)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract3_const_1"))
        return subtract_output_1


class Subtract4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, subtract_input_0, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, subtract_input_1)
        return subtract_output_1


class Subtract5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract5_const_0", shape=(5880, 2), dtype=torch.float32, use_random_value=True)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract5_const_0"), subtract_input_1)
        return subtract_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Subtract0,
        [((2, 1, 1, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Subtract0, [((2, 1, 7, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Subtract0, [((1, 1, 1, 204), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Subtract0, [((1, 1, 1, 201), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (
        Subtract0,
        [((1, 1, 1, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_base_v1_token_cls",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
            ]
        },
    ),
    (
        Subtract0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart",
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_opt_350m_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
                "pt_xglm_564M",
            ]
        },
    ),
    (
        Subtract1,
        [((1, 128), torch.int32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
            ]
        },
    ),
    (
        Subtract0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
            ]
        },
    ),
    (Subtract1, [((1, 384), torch.int32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Subtract0, [((1, 12, 384, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Subtract0, [((1, 1, 1, 256), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (
        Subtract1,
        [((1, 1, 256, 256), torch.int32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Subtract2,
        [((1, 32), torch.float32)],
        {
            "model_name": [
                "pt_opt_125m_seq_cls",
                "pt_opt_1_3b_seq_cls",
                "pt_opt_1_3b_qa",
                "pt_opt_350m_qa",
                "pt_opt_125m_qa",
                "pt_opt_350m_seq_cls",
            ]
        },
    ),
    (
        Subtract0,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_opt_125m_seq_cls",
                "pt_opt_1_3b_seq_cls",
                "pt_opt_1_3b_qa",
                "pt_opt_350m_qa",
                "pt_opt_125m_qa",
                "pt_opt_350m_seq_cls",
            ]
        },
    ),
    (
        Subtract3,
        [((1,), torch.int32)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_1_3b_seq_cls", "pt_opt_350m_seq_cls"]},
    ),
    (
        Subtract2,
        [((1, 256), torch.float32)],
        {"model_name": ["pt_opt_350m_causal_lm", "pt_opt_125m_causal_lm", "pt_opt_1_3b_causal_lm"]},
    ),
    (
        Subtract4,
        [((1024, 72), torch.float32), ((1024, 72), torch.float32)],
        {"model_name": ["nbeats_generic", "nbeats_trend", "nbeats_seasonality"]},
    ),
    (
        Subtract5,
        [((1, 5880, 2), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (
        Subtract4,
        [((1, 5880, 2), torch.float32), ((1, 5880, 2), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
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
