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
        [((2, 1, 1, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Abs0,
        [((2, 1, 7, 7), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Abs0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Abs0,
        [((1, 12, 384, 384), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Abs0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Abs0,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Abs")

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
