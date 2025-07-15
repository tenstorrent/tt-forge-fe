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
        self.add_constant("advindex0_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, advindex_input_0):
        advindex_output_1 = forge.op.AdvIndex("", advindex_input_0, self.get_constant("advindex0_const_1"))
        return advindex_output_1


class Advindex1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, advindex_input_0, advindex_input_1):
        advindex_output_1 = forge.op.AdvIndex("", advindex_input_0, advindex_input_1)
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


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Advindex0,
        [((1, 2), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 0,
        },
    ),
    (
        Advindex1,
        [((169, 3), torch.bfloat16), ((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 168,
        },
    ),
    (
        Advindex1,
        [((169, 6), torch.bfloat16), ((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 168,
        },
    ),
    (
        Advindex1,
        [((169, 12), torch.bfloat16), ((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 168,
        },
    ),
    (
        Advindex1,
        [((169, 24), torch.bfloat16), ((2401,), torch.int64)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 168,
        },
    ),
    (
        Advindex2,
        [((732, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 731,
        },
    ),
    (
        Advindex3,
        [((225, 3), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "max_int": 224},
    ),
    (
        Advindex3,
        [((225, 6), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "max_int": 224},
    ),
    (
        Advindex3,
        [((225, 12), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "max_int": 224},
    ),
    (
        Advindex3,
        [((225, 24), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "max_int": 224},
    ),
    (
        Advindex2,
        [((732, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 731,
        },
    ),
    (
        Advindex1,
        [((32, 2), torch.float32), ((1,), torch.int32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "max_int": 31},
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
