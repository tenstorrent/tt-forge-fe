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


class Sine0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, sine_input_0):
        sine_output_1 = forge.op.Sine("", sine_input_0)
        return sine_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Sine0,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Sine0, [((1, 35, 64), torch.float32)], {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99}),
    (Sine0, [((1, 588, 128), torch.float32)], {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99}),
    (
        Sine0,
        [((1, 12, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Sine0, [((1, 29, 128), torch.float32)], {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Sine0,
        [((1, 522, 256), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sine0,
        [((1, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_clm_hf", "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Sine0,
        [((1, 31, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_qwen_v3_embedding_4b_sentence_embed_gen_hf",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sine0,
        [((1, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Sine0,
        [((1, 11, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Sine0,
        [((1, 128, 128), torch.float32)],
        {"model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf", "pt_qwen_v3_4b_clm_hf"], "pcc": 0.99},
    ),
    (
        Sine0,
        [((1, 5, 32), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (Sine0, [((1, 35, 128), torch.float32)], {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99}),
    (Sine0, [((1, 29, 64), torch.float32)], {"model_names": ["pt_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99}),
    (Sine0, [((1, 39, 64), torch.float32)], {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99}),
    (Sine0, [((1, 334, 32), torch.float32)], {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (Sine0, [((1, 596, 128), torch.float32)], {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99}),
    (
        Sine0,
        [((1, 12, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
                "pt_phi4_microsoft_phi_4_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Sine0, [((1, 256, 96), torch.float32)], {"model_names": ["pt_phi3_5_mini_instruct_clm_hf"], "pcc": 0.99}),
    (
        Sine0,
        [((2, 160), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sine0,
        [((1, 25, 34, 64), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sine0,
        [((1, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_llama_3_2_1b_clm_hf", "pt_llama3_llama_3_2_1b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (Sine0, [((1, 1063, 128), torch.float32)], {"model_names": ["pt_deepseek_7b_instruct_qa_hf"], "pcc": 0.99}),
    (
        Sine0,
        [((1, 356, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (Sine0, [((1, 512, 256), torch.float32)], {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99}),
    (
        Sine0,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sine0,
        [((1, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sine0,
        [((1, 5, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sine0,
        [((1, 13, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Sine0, [((1, 5, 128), torch.float32)], {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99}),
    (Sine0, [((1, 13, 128), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99}),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Sine")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.get("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ["pcc"]:
            continue
        elif metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad=training_test)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(),
            dtype=parameter.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(),
            dtype=constant.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=training_test)

    verify(
        inputs,
        framework_model,
        compiled_model,
        with_backward=training_test,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
