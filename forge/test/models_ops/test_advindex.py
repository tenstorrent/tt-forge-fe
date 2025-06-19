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


class Advindex0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, advindex_input_0, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", advindex_input_0, advindex_input_1)
        return advindex_output_1


class Advindex1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("advindex1_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, advindex_input_0):
        advindex_output_1 = forge.op.AdvIndex("", advindex_input_0, self.get_constant("advindex1_const_1"))
        return advindex_output_1


class Advindex2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("advindex2_const_1", shape=(38809,), dtype=torch.int64)

    def forward(self, advindex_input_0):
        advindex_output_1 = forge.op.AdvIndex("", advindex_input_0, self.get_constant("advindex2_const_1"))
        return advindex_output_1


class Advindex3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("advindex3_const_1", shape=(4096,), dtype=torch.int64)

    def forward(self, advindex_input_0):
        advindex_output_1 = forge.op.AdvIndex("", advindex_input_0, self.get_constant("advindex3_const_1"))
        return advindex_output_1


class Advindex4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "advindex4.weight_0",
            forge.Parameter(*(169, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", self.get_parameter("advindex4.weight_0"), advindex_input_1)
        return advindex_output_1


class Advindex5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "advindex5.weight_0",
            forge.Parameter(*(169, 6), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", self.get_parameter("advindex5.weight_0"), advindex_input_1)
        return advindex_output_1


class Advindex6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "advindex6.weight_0",
            forge.Parameter(*(169, 12), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", self.get_parameter("advindex6.weight_0"), advindex_input_1)
        return advindex_output_1


class Advindex7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "advindex7.weight_0",
            forge.Parameter(*(169, 24), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", self.get_parameter("advindex7.weight_0"), advindex_input_1)
        return advindex_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Advindex0,
        [((448, 512), torch.float32), ((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Advindex0,
        [((7, 2), torch.float32), ((1,), torch.int64)],
        {"model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99, "max_int": 6},
    ),
    (
        Advindex1,
        [((1, 1, 1024, 72), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_seasionality_basis_clm_hf",
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 0,
        },
    ),
    (
        Advindex2,
        [((732, 12), torch.bfloat16)],
        {"model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"], "pcc": 0.99, "max_int": 731},
    ),
    (
        Advindex2,
        [((732, 16), torch.bfloat16)],
        {"model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"], "pcc": 0.99, "max_int": 731},
    ),
    (
        Advindex0,
        [((32, 2), torch.float32), ((1,), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Advindex0,
        [((169, 3), torch.bfloat16), ((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "max_int": 168,
        },
    ),
    (
        Advindex0,
        [((169, 6), torch.bfloat16), ((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "max_int": 168,
        },
    ),
    (
        Advindex0,
        [((169, 12), torch.bfloat16), ((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "max_int": 168,
        },
    ),
    (
        Advindex0,
        [((169, 24), torch.bfloat16), ((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "max_int": 168,
        },
    ),
    (
        Advindex0,
        [((169, 4), torch.bfloat16), ((2401,), torch.int64)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "max_int": 168},
    ),
    (
        Advindex0,
        [((169, 8), torch.bfloat16), ((2401,), torch.int64)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "max_int": 168},
    ),
    (
        Advindex0,
        [((169, 16), torch.bfloat16), ((2401,), torch.int64)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "max_int": 168},
    ),
    (
        Advindex0,
        [((169, 32), torch.bfloat16), ((2401,), torch.int64)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "max_int": 168},
    ),
    (
        Advindex0,
        [((448, 384), torch.float32), ((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Advindex1,
        [((1, 2), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 0,
        },
    ),
    (
        Advindex0,
        [((448, 1280), torch.float32), ((1, 2), torch.int64)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
                "pt_whisper_openai_whisper_large_v3_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 447,
        },
    ),
    (
        Advindex3,
        [((225, 3), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "max_int": 224,
        },
    ),
    (
        Advindex3,
        [((225, 6), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "max_int": 224,
        },
    ),
    (
        Advindex3,
        [((225, 12), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "max_int": 224,
        },
    ),
    (
        Advindex3,
        [((225, 24), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "max_int": 224,
        },
    ),
    (
        Advindex3,
        [((225, 4), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "max_int": 224},
    ),
    (
        Advindex3,
        [((225, 8), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "max_int": 224},
    ),
    (
        Advindex3,
        [((225, 16), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "max_int": 224},
    ),
    (
        Advindex3,
        [((225, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "max_int": 224},
    ),
    (
        Advindex0,
        [((448, 768), torch.float32), ((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Advindex0,
        [((4, 2), torch.float32), ((1,), torch.int64)],
        {"model_names": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99, "max_int": 3},
    ),
    (
        Advindex0,
        [((256, 2), torch.float32), ((1,), torch.int64)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "max_int": 255},
    ),
    (
        Advindex4,
        [((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 168,
        },
    ),
    (
        Advindex5,
        [((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 168,
        },
    ),
    (
        Advindex6,
        [((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 168,
        },
    ),
    (
        Advindex7,
        [((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 168,
        },
    ),
    (
        Advindex0,
        [((448, 1024), torch.float32), ((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("AdvIndex")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")
    max_int = metadata.pop("max_int")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

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

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
