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
        self.add_constant("subtract1_const_0", shape=(1,), dtype=torch.int32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract1_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract2_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract2_const_1"))
        return subtract_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Subtract0,
        [((1, 1, 1, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract1,
        [((1, 128), torch.int32)],
        {"model_name": ["pt_distilbert_distilbert_base_uncased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 12, 128, 128), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_uncased_mlm_hf"], "pcc": 0.99},
    ),
    (Subtract0, [((1, 1, 1, 256), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Subtract2, [((1, 256), torch.int64)], {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_forge_property):
    record_forge_property("framework_op_name", "Subtract")

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
