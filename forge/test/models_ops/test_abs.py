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


class Abs0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, abs_input_0):
        abs_output_1 = forge.op.Abs("", abs_input_0)
        return abs_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Abs0,
        [((1, 1, 256, 256), torch.float32)],
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
        },
    ),
    (
        Abs0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Abs0,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Abs0,
        [((2, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Abs0,
        [((1, 12, 384, 384), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (Abs0, [((1, 1, 7, 7), torch.float32)], {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99}),
    (
        Abs0,
        [((64, 3, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Abs0,
        [((16, 6, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Abs0,
        [((4, 12, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Abs0,
        [((1, 24, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Abs0,
        [((2, 1, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Abs")

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
