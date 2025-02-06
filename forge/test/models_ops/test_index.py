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


class Index0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=128, stride=1)
        return index_output_1


class Index1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=256, stride=1)
        return index_output_1


class Index2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1024, stride=1)
        return index_output_1


class Index3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=1792, stride=1)
        return index_output_1


class Index4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2304, stop=2560, stride=1)
        return index_output_1


class Index5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=512, stop=768, stride=1)
        return index_output_1


class Index6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1280, stop=1536, stride=1)
        return index_output_1


class Index7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2048, stop=2304, stride=1)
        return index_output_1


class Index8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2816, stop=3072, stride=1)
        return index_output_1


class Index9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=256, stop=512, stride=1)
        return index_output_1


class Index10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1024, stop=1280, stride=1)
        return index_output_1


class Index11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1792, stop=2048, stride=1)
        return index_output_1


class Index12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2560, stop=2816, stride=1)
        return index_output_1


class Index13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=1)
        return index_output_1


class Index14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=64, stride=1)
        return index_output_1


class Index15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=32, stride=2)
        return index_output_1


class Index16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=2)
        return index_output_1


class Index17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=1, stride=1)
        return index_output_1


class Index18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=2, stride=1)
        return index_output_1


class Index19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=1, stride=1)
        return index_output_1


class Index20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=2, stride=1)
        return index_output_1


class Index21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=768, stride=1)
        return index_output_1


class Index22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1536, stride=1)
        return index_output_1


class Index23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=2304, stride=1)
        return index_output_1


class Index24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=768, stride=1)
        return index_output_1


class Index25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=768, stop=1536, stride=1)
        return index_output_1


class Index26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1536, stop=2304, stride=1)
        return index_output_1


class Index27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=256, stride=1)
        return index_output_1


class Index28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=258, stride=1)
        return index_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Index0,
        [((1, 512), torch.int64)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
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
        },
    ),
    (
        Index1,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index2,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index3,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index4,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index5,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index6,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index7,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index8,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index9,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index10,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index11,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index12,
        [((3072, 1024), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index13,
        [((1, 256, 16, 64), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index14,
        [((1, 256, 16, 64), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index15,
        [((1, 256, 16, 32), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index16,
        [((1, 256, 16, 32), torch.float32)],
        {"model_name": ["pt_codegen_salesforce_codegen_350m_mono_clm_hf"], "pcc": 0.99},
    ),
    (
        Index17,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Index17,
        [((2, 768), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Index18,
        [((2, 768), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Index19,
        [((2,), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Index20,
        [((2,), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
        },
    ),
    (Index21, [((2304, 768), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Index22, [((2304, 768), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Index23, [((2304, 768), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Index24, [((2304,), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Index25, [((2304,), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Index26, [((2304,), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Index1, [((1, 1, 1024, 1024), torch.bool)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Index27, [((1, 1, 256, 1024), torch.bool)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (
        Index1,
        [((1, 1, 2048, 2048), torch.bool)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Index27,
        [((1, 1, 256, 2048), torch.bool)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (Index14, [((1, 16, 6, 64), torch.float32)], {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (Index13, [((1, 16, 6, 64), torch.float32)], {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (Index0, [((1, 514), torch.int64)], {"model_name": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99}),
    (Index28, [((2050, 1024), torch.float32)], {"model_name": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99}),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_forge_property):
    record_forge_property("framework_op_name", "Index")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        record_forge_property(metadata_name, metadata_value)

    integer_tensor_high_value = 1000
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
