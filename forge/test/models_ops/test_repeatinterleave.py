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


class Repeatinterleave0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=0)
        return repeatinterleave_output_1


class Repeatinterleave1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=1)
        return repeatinterleave_output_1


class Repeatinterleave2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=256, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=4, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=32, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=7, dim=2)
        return repeatinterleave_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Repeatinterleave0,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave2,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "256", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 8, 1, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave5,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "32", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave1,
        [((2, 1, 1, 7), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave6,
        [((2, 1, 1, 7), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 8, 1, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 14), torch.int64)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 384), torch.int64)],
        {
            "model_names": ["pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave1,
        [((2, 1, 1, 13), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave3,
        [((2, 1, 1, 13), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("RepeatInterleave")

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
