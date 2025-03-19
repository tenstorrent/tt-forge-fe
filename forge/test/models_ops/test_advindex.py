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
        self.add_parameter(
            "advindex2.weight_0",
            forge.Parameter(*(169, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", self.get_parameter("advindex2.weight_0"), advindex_input_1)
        return advindex_output_1


class Advindex3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "advindex3.weight_0",
            forge.Parameter(*(169, 6), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", self.get_parameter("advindex3.weight_0"), advindex_input_1)
        return advindex_output_1


class Advindex4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "advindex4.weight_0",
            forge.Parameter(*(169, 12), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", self.get_parameter("advindex4.weight_0"), advindex_input_1)
        return advindex_output_1


class Advindex5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "advindex5.weight_0",
            forge.Parameter(*(169, 24), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", self.get_parameter("advindex5.weight_0"), advindex_input_1)
        return advindex_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Advindex0,
        [((448, 1024), torch.float32), ((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Advindex0,
        [((448, 1280), torch.float32), ((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Advindex0,
        [((448, 384), torch.float32), ((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Advindex0,
        [((448, 512), torch.float32), ((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Advindex0,
        [((448, 768), torch.float32), ((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Advindex0,
        [((448, 1280), torch.float32), ((1, 2), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Advindex1,
        [((1, 2), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 0,
        },
    ),
    (
        Advindex0,
        [((32, 2), torch.float32), ((1,), torch.int64)],
        {
            "model_name": [
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Advindex1,
        [((1, 1, 1024, 72), torch.float32)],
        {
            "model_name": [
                "pt_nbeats_seasionality_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_generic_basis_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 0,
        },
    ),
    (
        Advindex2,
        [((2401,), torch.int64)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99, "max_int": 168},
    ),
    (
        Advindex3,
        [((2401,), torch.int64)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99, "max_int": 168},
    ),
    (
        Advindex4,
        [((2401,), torch.int64)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99, "max_int": 168},
    ),
    (
        Advindex5,
        [((2401,), torch.int64)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99, "max_int": 168},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder.record_op_name("AdvIndex")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")
    max_int = metadata.pop("max_int")

    for metadata_name, metadata_value in metadata.items():
        forge_property_recorder("tags." + str(metadata_name), metadata_value)

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
