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
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=128, dim=2)
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
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=256, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=6, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=35, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=29, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave9(ForgeModule):
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
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave2,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "128", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
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
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
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
        [((1, 16, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave5,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "256", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 6), torch.int64)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 6), torch.int64)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave6,
        [((1, 1, 1, 6), torch.int64)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "6", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave6,
        [((1, 2, 1, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "6", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 35), torch.int64)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 35), torch.int64)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave7,
        [((1, 1, 1, 35), torch.int64)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "35", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave6,
        [((1, 2, 1, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "6", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 29), torch.int64)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 29), torch.int64)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave8,
        [((1, 1, 1, 29), torch.int64)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "29", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave9,
        [((1, 2, 1, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "7", "dim": "2"}},
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
