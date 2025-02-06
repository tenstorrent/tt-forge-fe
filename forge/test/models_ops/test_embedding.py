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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
import pytest


class Embedding0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, embedding_input_0, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, embedding_input_1)
        return embedding_output_1


class Embedding1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding1_const_0", shape=(1, 256), dtype=torch.int64)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding1_const_0"), embedding_input_1)
        return embedding_output_1


class Embedding2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding2_const_0", shape=(1, 128), dtype=torch.int64)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding2_const_0"), embedding_input_1)
        return embedding_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Embedding0,
        [((1, 128), torch.int64), ((30000, 128), torch.bfloat16)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "integer_tensor_high_value": 29999,
        },
    ),
    (
        Embedding0,
        [((1, 128), torch.int64), ((2, 128), torch.bfloat16)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "integer_tensor_high_value": 1,
        },
    ),
    (
        Embedding0,
        [((1, 128), torch.int64), ((512, 128), torch.bfloat16)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "integer_tensor_high_value": 511,
        },
    ),
    (
        Embedding0,
        [((1, 256), torch.int64), ((50265, 1024), torch.bfloat16)],
        {
            "model_name": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "integer_tensor_high_value": 50264,
        },
    ),
    (
        Embedding1,
        [((1026, 1024), torch.bfloat16)],
        {"model_name": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99, "integer_tensor_high_value": 1025},
    ),
    (
        Embedding0,
        [((1, 256), torch.int32), ((51200, 1024), torch.bfloat16)],
        {
            "model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"],
            "pcc": 0.99,
            "integer_tensor_high_value": 51199,
        },
    ),
    (
        Embedding0,
        [((1, 128), torch.int64), ((30522, 768), torch.bfloat16)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "integer_tensor_high_value": 30521,
        },
    ),
    (
        Embedding0,
        [((1, 128), torch.int64), ((512, 768), torch.bfloat16)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "integer_tensor_high_value": 511,
        },
    ),
    (
        Embedding0,
        [((1, 128), torch.int64), ((2, 768), torch.bfloat16)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "integer_tensor_high_value": 1,
        },
    ),
    (
        Embedding0,
        [((1, 256), torch.int64), ((50257, 768), torch.bfloat16)],
        {
            "model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "integer_tensor_high_value": 50256,
        },
    ),
    (
        Embedding1,
        [((1024, 768), torch.bfloat16)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99, "integer_tensor_high_value": 1023},
    ),
    (
        Embedding1,
        [((2048, 768), torch.bfloat16)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99, "integer_tensor_high_value": 2047},
    ),
    (
        Embedding0,
        [((1, 256), torch.int64), ((50272, 768), torch.bfloat16)],
        {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "integer_tensor_high_value": 50271},
    ),
    (
        Embedding0,
        [((1, 256), torch.int64), ((2050, 768), torch.bfloat16)],
        {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "integer_tensor_high_value": 2049},
    ),
    (
        Embedding0,
        [((1, 6), torch.int64), ((151936, 1024), torch.bfloat16)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "integer_tensor_high_value": 151935},
    ),
    (
        Embedding0,
        [((1, 128), torch.int64), ((250002, 768), torch.bfloat16)],
        {"model_name": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99, "integer_tensor_high_value": 250001},
    ),
    (
        Embedding0,
        [((1, 128), torch.int64), ((1, 768), torch.bfloat16)],
        {"model_name": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99, "integer_tensor_high_value": 0},
    ),
    (
        Embedding0,
        [((1, 128), torch.int64), ((514, 768), torch.bfloat16)],
        {"model_name": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99, "integer_tensor_high_value": 513},
    ),
    (
        Embedding0,
        [((1, 128), torch.int64), ((30528, 768), torch.bfloat16)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "integer_tensor_high_value": 30527,
        },
    ),
    (
        Embedding2,
        [((2, 768), torch.bfloat16)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "integer_tensor_high_value": 1,
        },
    ),
    (
        Embedding0,
        [((1, 256), torch.int64), ((256008, 1024), torch.bfloat16)],
        {"model_name": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99, "integer_tensor_high_value": 256007},
    ),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_forge_property):
    record_forge_property("framework_op_name", "Embedding")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")
    integer_tensor_high_value = metadata.pop("integer_tensor_high_value")

    for metadata_name, metadata_value in metadata.items():
        record_forge_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, integer_tensor_high_value=integer_tensor_high_value)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(),
            dtype=parameter.pt_data_format,
            integer_tensor_high_value=integer_tensor_high_value,
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(),
            dtype=constant.pt_data_format,
            integer_tensor_high_value=integer_tensor_high_value,
        )
        framework_model.set_constant(name, constant_tensor)

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
