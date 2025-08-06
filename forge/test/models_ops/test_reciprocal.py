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


class Reciprocal0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reciprocal_input_0):
        reciprocal_output_1 = forge.op.Reciprocal("", reciprocal_input_0)
        return reciprocal_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reciprocal0,
        [((1, 44, 1), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 39, 1), torch.float32)],
        {
            "model_names": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 522, 1), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 128, 1), torch.float32)],
        {
            "model_names": [
                "pt_gemma_google_gemma_2b_text_gen_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 107, 1), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 4, 1), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 10, 1), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 8, 1), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 135, 1), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 256, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 5, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 13, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_qwen_v2_qwen_qwen2_7b_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 12, 1), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 35, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 29, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 38, 1), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 513, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 61, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((2, 13, 1), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Reciprocal0, [((1, 6, 1), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    pytest.param(
        (
            Reciprocal0,
            [((1, 1, 1), torch.float32)],
            {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Reciprocal0,
        [((1, 588, 1), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (Reciprocal0, [((1, 1, 1, 64), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 256), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 128), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 512), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 1024), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 2048), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Reciprocal")

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
