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
        [((1, 1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Abs0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Abs0, [((1, 1, 6, 6), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Abs0,
        [((1, 1, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Abs0,
        [((1, 1, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Abs0, [((64, 3, 64, 32), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (Abs0, [((16, 6, 64, 32), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (Abs0, [((4, 12, 64, 32), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (Abs0, [((1, 24, 64, 32), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
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
